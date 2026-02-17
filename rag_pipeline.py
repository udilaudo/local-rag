"""
RAG Pipeline - Cuore del sistema.

Questo modulo gestisce tutto il flusso RAG:
1. Carica i PDF dalla cartella documents/
2. Divide il testo in chunk (pezzi) con sovrapposizione
3. Crea embeddings vettoriali per ogni chunk
4. Salva tutto in ChromaDB (database vettoriale)
5. Quando arriva una domanda, cerca i chunk più simili
6. Passa i chunk + domanda a Ollama per generare la risposta
"""

import os
import glob
import shutil
from typing import Optional

# LangChain: framework per orchestrare la pipeline
# NOTA: nelle versioni recenti di LangChain (v1.x), i moduli sono stati
# spostati in pacchetti separati (langchain_community, langchain_text_splitters, ecc.)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# Importa la configurazione centralizzata
import config


# Template del prompt: istruzioni chiare per il modello
# {context} verrà sostituito con i chunk trovati
# {question} verrà sostituito con la domanda dell'utente
TEMPLATE_PROMPT = """Sei un assistente esperto che risponde a domande basandosi ESCLUSIVAMENTE sui documenti forniti.

REGOLE IMPORTANTI:
1. Rispondi SOLO usando le informazioni presenti nel contesto qui sotto
2. Se il contesto non contiene informazioni sufficienti, dillo chiaramente
3. Cita sempre le fonti: indica da quale documento e pagina proviene ogni informazione
4. Rispondi in italiano
5. Sii preciso e conciso

CONTESTO (estratto dai documenti):
{context}

DOMANDA: {question}

RISPOSTA (con citazione delle fonti):"""


def carica_pdf(cartella: str = config.DOCUMENTS_DIR) -> list:
    """
    Carica tutti i PDF da una cartella.

    Ogni pagina di ogni PDF diventa un "Document" di LangChain,
    con il testo della pagina e i metadati (nome file, numero pagina).

    Args:
        cartella: percorso alla cartella con i PDF

    Returns:
        lista di Document (uno per ogni pagina di ogni PDF)
    """
    documenti = []  # qui accumulo tutti i documenti caricati

    # Trovo tutti i file .pdf nella cartella
    pattern = os.path.join(cartella, "*.pdf")
    file_pdf = glob.glob(pattern)

    if not file_pdf:
        print(f"⚠️ Nessun PDF trovato in: {cartella}")
        return documenti

    for percorso_pdf in file_pdf:
        nome_file = os.path.basename(percorso_pdf)
        print(f"📄 Caricamento: {nome_file}")

        try:
            # PyPDFLoader estrae il testo da ogni pagina del PDF
            loader = PyPDFLoader(percorso_pdf)
            pagine = loader.load()

            # Aggiungo il nome del file ai metadati di ogni pagina
            # così dopo posso citare la fonte nella risposta
            for pagina in pagine:
                pagina.metadata["source_filename"] = nome_file

            documenti.extend(pagine)
            print(f"   ✅ Caricate {len(pagine)} pagine")

        except Exception as e:
            print(f"   ❌ Errore nel caricamento di {nome_file}: {e}")

    print(f"\n📚 Totale: {len(documenti)} pagine da {len(file_pdf)} PDF")
    return documenti


def chunking(documenti: list,
             chunk_size: int = config.CHUNK_SIZE,
             chunk_overlap: int = config.CHUNK_OVERLAP) -> list:
    """
    Divide i documenti in chunk (pezzi) più piccoli.

    Perché fare chunking?
    - I modelli LLM hanno un limite di contesto (quanti token possono leggere)
    - Chunk più piccoli = ricerca più precisa (il vettore rappresenta meglio il contenuto)
    - L'overlap assicura che nessun concetto venga "tagliato a metà"

    RecursiveCharacterTextSplitter prova a dividere prima per paragrafi,
    poi per frasi, poi per parole - mantiene il testo il più coerente possibile.

    Args:
        documenti: lista di Document da dividere
        chunk_size: dimensione massima di ogni chunk (in caratteri)
        chunk_overlap: sovrapposizione tra chunk consecutivi

    Returns:
        lista di Document (chunk), ognuno con i metadati originali
    """
    # Creo lo splitter con la strategia "ricorsiva"
    # I separatori sono provati in ordine: prima paragrafi, poi frasi, poi spazi
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # dal più grande al più piccolo
        length_function=len,  # misura la lunghezza in caratteri
    )

    # Divido tutti i documenti in chunk
    chunks = splitter.split_documents(documenti)
    print(f"✂️ Creati {len(chunks)} chunk (size={chunk_size}, overlap={chunk_overlap})")

    return chunks


def _crea_embeddings() -> HuggingFaceEmbeddings:
    """
    Crea la funzione di embedding usando sentence-transformers.

    Funzione di utilità usata sia per creare che per caricare il vector store,
    così il modello di embedding è sempre lo stesso.

    Returns:
        oggetto HuggingFaceEmbeddings pronto per trasformare testo in vettori
    """
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},  # usa la CPU (funziona ovunque)
    )


def crea_vector_store(chunks: list,
                      persist_directory: str = config.CHROMA_DB_DIR) -> Chroma:
    """
    Crea il database vettoriale da una lista di chunk.

    Cosa fa:
    1. Prende ogni chunk di testo
    2. Lo trasforma in un vettore numerico (embedding) usando sentence-transformers
    3. Salva vettore + testo + metadati in ChromaDB

    ChromaDB è un database vettoriale che permette di cercare
    per "similarità semantica" - cioè trova testi con significato simile,
    non solo con le stesse parole esatte.

    Args:
        chunks: lista di Document (chunk) da indicizzare
        persist_directory: dove salvare il database su disco

    Returns:
        oggetto Chroma (il vector store pronto per le ricerche)
    """
    print(f"🧮 Creazione embeddings con modello: {config.EMBEDDING_MODEL}")
    print("   (il primo avvio scarica il modello, ~80MB, poi è tutto locale)")

    # Creo la funzione di embedding usando sentence-transformers
    # Questo modello trasforma testo -> vettore di 384 dimensioni
    embeddings = _crea_embeddings()

    # Creo il vector store ChromaDB e ci salvo tutti i chunk
    # persist_directory = dove salvare su disco (così non devo ri-indicizzare ogni volta)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=config.COLLECTION_NAME,
    )

    print(f"💾 Vector store creato e salvato in: {persist_directory}")
    print(f"   Contiene {len(chunks)} vettori")

    return vector_store


def carica_vector_store(persist_directory: str = config.CHROMA_DB_DIR) -> Optional[Chroma]:
    """
    Carica un vector store esistente da disco.

    Utile per non dover ri-indicizzare i documenti ogni volta che
    si riavvia l'applicazione. Se il database non esiste, ritorna None.

    Args:
        persist_directory: dove cercare il database salvato

    Returns:
        oggetto Chroma se trovato, None altrimenti
    """
    # Verifico se la cartella del database esiste
    if not os.path.exists(persist_directory):
        print("⚠️ Nessun vector store trovato su disco.")
        return None

    print(f"📂 Caricamento vector store da: {persist_directory}")

    # Ricreo la stessa funzione di embedding (serve per le ricerche)
    embeddings = _crea_embeddings()

    # Carico il database esistente
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=config.COLLECTION_NAME,
    )

    # Verifico che ci siano effettivamente dei dati
    conteggio = vector_store._collection.count()
    if conteggio == 0:
        print("⚠️ Vector store vuoto, serve re-indicizzazione.")
        return None

    print(f"✅ Vector store caricato: {conteggio} vettori")
    return vector_store


def crea_catena_rag(vector_store: Chroma) -> dict:
    """
    Crea i componenti della catena RAG: retriever + LLM + prompt.

    Invece di usare la vecchia classe RetrievalQA (rimossa in LangChain v1.x),
    prepariamo i singoli componenti e li usiamo direttamente in fai_domanda().
    Questo approccio è più semplice, più trasparente e più facile da debuggare.

    La catena funziona così:
    1. Riceve una domanda dall'utente
    2. Il retriever cerca i chunk più simili nel vector store
    3. Il prompt combina: istruzioni + chunk trovati + domanda
    4. L'LLM (Ollama) genera la risposta

    Args:
        vector_store: il database vettoriale con i chunk indicizzati

    Returns:
        dizionario con i componenti: retriever, llm, prompt
    """
    # Configuro il retriever: cerca i top-K chunk più simili alla domanda
    retriever = vector_store.as_retriever(
        search_type="similarity",  # ricerca per similarità coseno
        search_kwargs={"k": config.TOP_K},  # quanti chunk recuperare
    )

    # Creo il modello LLM (Ollama locale)
    llm = OllamaLLM(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.OLLAMA_TEMPERATURE,
    )

    # Creo il template del prompt
    prompt = PromptTemplate(
        template=TEMPLATE_PROMPT,
        input_variables=["context", "question"],
    )

    print("🔗 Catena RAG creata e pronta!")

    # Ritorno i componenti come dizionario
    # Li useremo in fai_domanda() per eseguire la pipeline step by step
    return {
        "retriever": retriever,
        "llm": llm,
        "prompt": prompt,
    }


def indicizza_documenti() -> Chroma:
    """
    Pipeline completa di indicizzazione: carica PDF → chunk → vector store.

    Questa funzione esegue tutto il processo di preparazione:
    1. Carica tutti i PDF dalla cartella documents/
    2. Li divide in chunk
    3. Crea gli embeddings e li salva in ChromaDB

    Returns:
        il vector store creato e popolato
    """
    print("=" * 50)
    print("🚀 INIZIO INDICIZZAZIONE DOCUMENTI")
    print("=" * 50)

    # Step 0: cancello il vecchio vector store se esiste
    # Questo evita che chunk di documenti eliminati restino nel database
    if os.path.exists(config.CHROMA_DB_DIR):
        shutil.rmtree(config.CHROMA_DB_DIR)
        print("🗑️ Vecchio indice cancellato")

    # Step 1: carica i PDF
    documenti = carica_pdf()
    if not documenti:
        raise ValueError("Nessun documento trovato! Metti dei PDF nella cartella documents/")

    # Step 2: dividi in chunk
    chunks = chunking(documenti)

    # Step 3: crea vector store con embeddings
    vector_store = crea_vector_store(chunks)

    print("=" * 50)
    print("✅ INDICIZZAZIONE COMPLETATA!")
    print("=" * 50)

    return vector_store


def fai_domanda(domanda: str, catena: dict) -> dict:
    """
    Fa una domanda alla pipeline RAG e restituisce la risposta con le fonti.

    Esegue la pipeline RAG passo per passo:
    1. Usa il retriever per trovare i chunk più simili alla domanda
    2. Combina i chunk nel prompt insieme alla domanda
    3. Manda il prompt all'LLM per generare la risposta

    Args:
        domanda: la domanda dell'utente in linguaggio naturale
        catena: dizionario con i componenti RAG (da crea_catena_rag())

    Returns:
        dizionario con:
        - "risposta": il testo generato dal modello
        - "fonti": lista di dict con info sui chunk usati (documento, pagina, testo)
    """
    print(f"\n❓ Domanda: {domanda}")
    print("🔍 Ricerca chunk rilevanti e generazione risposta...")

    # Estraggo i componenti della catena
    retriever = catena["retriever"]
    llm = catena["llm"]
    prompt = catena["prompt"]

    # Step 1: cerco i chunk più simili alla domanda nel vector store
    # Il retriever usa la similarità coseno tra l'embedding della domanda
    # e gli embeddings dei chunk salvati
    documenti_trovati = retriever.invoke(domanda)

    # Step 2: combino il testo dei chunk trovati in un unico "contesto"
    # Questo contesto verrà inserito nel prompt per l'LLM
    contesto = "\n\n---\n\n".join(
        [doc.page_content for doc in documenti_trovati]
    )

    # Step 3: creo il prompt finale con contesto + domanda
    prompt_finale = prompt.format(context=contesto, question=domanda)

    # Step 4: mando il prompt all'LLM (Ollama) e ottengo la risposta
    risposta = llm.invoke(prompt_finale)

    # Estraggo le informazioni sulle fonti usate
    # Così l'utente sa da dove viene ogni informazione
    fonti = []
    for doc in documenti_trovati:
        fonte = {
            "documento": doc.metadata.get("source_filename", "sconosciuto"),
            "pagina": doc.metadata.get("page", "n/a"),
            "testo_chunk": doc.page_content[:300] + "..."  # mostro solo i primi 300 char
        }
        fonti.append(fonte)

    print("✅ Risposta generata!")

    return {
        "risposta": risposta,
        "fonti": fonti,
    }
