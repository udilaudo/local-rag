"""
App Streamlit - Interfaccia utente per la RAG Pipeline.

Questa è l'interfaccia grafica del progetto.
L'utente può:
- Indicizzare i documenti PDF (sidebar)
- Fare domande sui documenti (area principale)
- Vedere le risposte con le fonti citate
"""

import os
import streamlit as st

# Importo le funzioni della pipeline RAG
from rag_pipeline import (
    indicizza_documenti,
    carica_vector_store,
    crea_catena_rag,
    fai_domanda,
)
import config


# === CONFIGURAZIONE PAGINA ===
# Queste impostazioni definiscono l'aspetto della pagina Streamlit
st.set_page_config(
    page_title="RAG Personale",
    page_icon="📚",
    layout="wide",  # usa tutta la larghezza dello schermo
)


def controlla_ollama() -> bool:
    """
    Verifica se Ollama è in esecuzione e raggiungibile.

    Prova a fare una richiesta HTTP al server Ollama.
    Se fallisce, significa che Ollama non è avviato.

    Returns:
        True se Ollama risponde, False altrimenti
    """
    import urllib.request

    try:
        # Provo a contattare il server Ollama
        urllib.request.urlopen(config.OLLAMA_BASE_URL, timeout=3)
        return True
    except Exception:
        return False


def lista_documenti() -> list:
    """
    Restituisce la lista dei file PDF nella cartella documents/.

    Returns:
        lista di nomi file PDF trovati
    """
    if not os.path.exists(config.DOCUMENTS_DIR):
        return []

    # Filtro solo i file che finiscono con .pdf (case insensitive)
    return [f for f in os.listdir(config.DOCUMENTS_DIR) if f.lower().endswith(".pdf")]


# === SIDEBAR ===
# La sidebar contiene i controlli: indicizzazione e lista documenti
with st.sidebar:
    st.header("⚙️ Gestione Documenti")

    # Mostro la lista dei PDF trovati
    pdf_trovati = lista_documenti()
    st.subheader(f"📄 Documenti trovati ({len(pdf_trovati)})")
    if pdf_trovati:
        for pdf in pdf_trovati:
            st.text(f"• {pdf}")
    else:
        st.warning("Nessun PDF trovato nella cartella documents/")

    st.divider()  # linea separatrice

    # Bottone per (re)indicizzare i documenti
    # Quando premuto, esegue l'intera pipeline di indicizzazione
    if st.button("🔄 Indicizza Documenti", type="primary", use_container_width=True):
        # Verifico che Ollama sia attivo prima di procedere
        if not controlla_ollama():
            st.error(
                "❌ Ollama non è in esecuzione!\n\n"
                "Avvialo con il comando:\n"
                "```\nollama serve\n```"
            )
        elif not pdf_trovati:
            st.error("❌ Nessun PDF trovato nella cartella documents/")
        else:
            # Eseguo l'indicizzazione con una barra di progresso
            with st.spinner(
                "📊 Indicizzazione in corso... (può richiedere qualche minuto)"
            ):
                try:
                    # Lancio la pipeline completa
                    vector_store = indicizza_documenti()

                    # Salvo il vector store nella sessione di Streamlit
                    # così resta disponibile tra le interazioni
                    st.session_state["vector_store"] = vector_store
                    st.session_state["catena"] = crea_catena_rag(vector_store)

                    st.success("✅ Indicizzazione completata!")
                except Exception as e:
                    st.error(f"❌ Errore durante l'indicizzazione: {e}")

    st.divider()

    # Info sullo stato del sistema
    st.subheader("📊 Stato Sistema")

    # Controllo se Ollama è attivo
    if controlla_ollama():
        st.success(f"✅ Ollama attivo ({config.OLLAMA_MODEL})")
    else:
        st.error("❌ Ollama non attivo")

    # Controllo se il vector store è caricato
    if "vector_store" in st.session_state:
        st.success("✅ Indice caricato")
    else:
        st.info("ℹ️ Indice non caricato")


# === AREA PRINCIPALE ===
# Titolo e descrizione
st.title("📚 RAG Pipeline - Paper Scientifici")
st.markdown(
    "Fai domande sui tuoi documenti PDF. "
    "Il sistema cerca le informazioni rilevanti e genera una risposta con le fonti."
)

# All'avvio, provo a caricare un vector store esistente
# (così non serve re-indicizzare se è già stato fatto prima)
if "vector_store" not in st.session_state:
    vector_store = carica_vector_store()
    if vector_store is not None:
        st.session_state["vector_store"] = vector_store
        st.session_state["catena"] = crea_catena_rag(vector_store)

# Campo per inserire la domanda
domanda = st.text_input(
    "🔍 Fai una domanda sui tuoi documenti:",
    placeholder="Es: Quali sono i risultati principali dello studio?",
)

# Quando l'utente preme Invio o clicca il bottone
if domanda:
    # Verifico che tutto sia pronto
    if "catena" not in st.session_state:
        st.warning(
            "⚠️ Devi prima indicizzare i documenti! " "Clicca il bottone nella sidebar."
        )
    elif not controlla_ollama():
        st.error("❌ Ollama non è in esecuzione! " "Avvialo con: `ollama serve`")
    else:
        # Tutto pronto: faccio la domanda alla pipeline RAG
        with st.spinner("🤔 Sto elaborando la risposta..."):
            try:
                risultato = fai_domanda(domanda, st.session_state["catena"])

                # Mostro la risposta principale
                st.subheader("💬 Risposta")
                st.markdown(risultato["risposta"])

                # Mostro le fonti usate in un expander (sezione espandibile)
                st.subheader("📖 Fonti")
                for i, fonte in enumerate(risultato["fonti"], 1):
                    with st.expander(
                        f"Fonte {i}: {fonte['documento']} - Pagina {fonte['pagina']}"
                    ):
                        # Mostro un estratto del chunk usato
                        st.text(fonte["testo_chunk"])

            except Exception as e:
                st.error(f"❌ Errore nella generazione della risposta: {e}")

# Footer con istruzioni
st.divider()
st.caption(
    "💡 **Come usare**: "
    "1) Metti i PDF nella cartella `documents/` → "
    "2) Avvia Ollama (`ollama serve`) → "
    "3) Clicca 'Indicizza Documenti' nella sidebar → "
    "4) Fai le tue domande!"
)
