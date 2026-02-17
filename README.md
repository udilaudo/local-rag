# Local RAG - Offline Personal Document Assistant

A fully offline Retrieval-Augmented Generation (RAG) pipeline for querying your personal PDF documents using natural language. Everything runs locally on your machine — no data is sent to external services.

## How It Works

1. **PDF Loading** — Reads all PDF files from the `documents/` folder
2. **Chunking** — Splits text into overlapping chunks for better search accuracy
3. **Embedding** — Converts chunks into vector representations using `all-MiniLM-L6-v2` (sentence-transformers)
4. **Vector Store** — Stores embeddings in ChromaDB for fast semantic search
5. **Question Answering** — When you ask a question, the system retrieves the most relevant chunks and generates an answer using a local LLM (Ollama), citing the sources

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) installed locally
- Internet connection **only during setup** (to download models)

## Setup (requires internet, one time only)

```bash
# 1. Create a conda environment (or use an existing one)
conda create -n personal python=3.12
conda activate personal

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download the embedding model locally (stored in models/)
python setup_offline.py

# 4. Pull the LLM model into Ollama
ollama pull llama3.2:3b
```

After this setup, the app works **completely offline** — no internet needed.

## Usage

```bash
# 1. Start the Ollama server
ollama serve

# 2. Place your PDFs in the documents/ folder, then run:
streamlit run app.py
```

In the web interface:
1. Click **"Indicizza Documenti"** in the sidebar to index your PDFs
2. Type a question in the text field and press Enter
3. The system will show the answer along with the source documents and pages

## Project Structure

```
local-rag/
├── app.py              # Streamlit web interface
├── rag_pipeline.py     # Core RAG logic (loading, chunking, embedding, querying)
├── config.py           # All configurable parameters in one place
├── setup_offline.py    # One-time setup script to download models locally
├── requirements.txt    # Python dependencies
├── documents/          # Put your PDF files here
├── models/             # Local embedding model (created by setup_offline.py)
└── chroma_db/          # Vector database (created automatically)
```

## Configuration

All parameters can be adjusted in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 1000 | Size of each text chunk (characters) |
| `CHUNK_OVERLAP` | 200 | Overlap between consecutive chunks |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model for embeddings |
| `OLLAMA_MODEL` | `llama3.2:3b` | Local LLM model via Ollama |
| `OLLAMA_TEMPERATURE` | 0.1 | Response creativity (0 = deterministic, 1 = creative) |
| `TOP_K` | 4 | Number of relevant chunks retrieved per question |

## Tech Stack

- **LangChain** — Pipeline orchestration
- **Ollama** — Local LLM inference
- **ChromaDB** — Vector database
- **Sentence-Transformers** — Local text embeddings
- **Streamlit** — Web UI
- **PyPDF** — PDF text extraction
