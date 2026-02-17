"""
Configurazione centralizzata per la RAG pipeline.
Tutti i parametri modificabili sono qui, così non devi cercarli nel codice.
"""

import os

# === PERCORSI ===
# Cartella dove mettere i PDF da indicizzare
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")

# Cartella dove ChromaDB salva il database vettoriale (creata automaticamente)
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# === CHUNKING ===
# Dimensione di ogni chunk di testo (in caratteri)
# 1000 è un buon compromesso: abbastanza grande da avere contesto,
# abbastanza piccolo da essere preciso nella ricerca
CHUNK_SIZE = 1000

# Sovrapposizione tra chunk consecutivi (in caratteri)
# L'overlap evita di "tagliare" concetti a metà tra due chunk
CHUNK_OVERLAP = 200

# === EMBEDDINGS ===
# Modello sentence-transformers per creare i vettori dai testi
# all-MiniLM-L6-v2 è leggero (~80MB) e veloce, ottimo per iniziare
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# === LLM (Ollama) ===
# URL del server Ollama locale
OLLAMA_BASE_URL = "http://localhost:11434"

# Modello Ollama da usare per generare le risposte
# llama3.2 è un modello 3B, veloce anche su Mac
OLLAMA_MODEL = "llama3.2:3b"

# Temperatura: 0 = risposte deterministiche, 1 = più creative
# Per RAG su documenti scientifici, meglio tenerla bassa
OLLAMA_TEMPERATURE = 0.1

# === RETRIEVAL ===
# Quanti chunk simili recuperare per ogni domanda
# 4 è un buon default: abbastanza contesto senza sovraccaricare il prompt
TOP_K = 4

# === NOME COLLEZIONE CHROMADB ===
# Nome della collezione nel database vettoriale
COLLECTION_NAME = "my-documents"
