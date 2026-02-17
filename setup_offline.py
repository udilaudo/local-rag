"""
Script di setup da eseguire UNA SOLA VOLTA con internet.

Scarica il modello di embedding (all-MiniLM-L6-v2) nella cartella locale
models/ del progetto, così l'app può funzionare completamente offline.

Uso:
    conda activate personal
    python setup_offline.py
"""

import os
from sentence_transformers import SentenceTransformer
import config

# Cartella locale dove salvare il modello (dentro il progetto)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def main():
    print("=" * 50)
    print("🔧 SETUP OFFLINE")
    print("=" * 50)

    # Creo la cartella models/ se non esiste
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Percorso dove salvare il modello di embedding
    model_path = os.path.join(MODELS_DIR, config.EMBEDDING_MODEL)

    if os.path.exists(model_path):
        print(f"✅ Modello già scaricato in: {model_path}")
    else:
        print(f"📥 Scaricamento modello: {config.EMBEDDING_MODEL}")
        print("   (circa 80MB, serve solo questa volta)")

        # Scarico il modello da HuggingFace e lo salvo in locale
        model = SentenceTransformer(config.EMBEDDING_MODEL)
        model.save(model_path)

        print(f"✅ Modello salvato in: {model_path}")

    print()
    print("=" * 50)
    print("✅ SETUP COMPLETATO!")
    print("   Ora puoi usare l'app completamente offline.")
    print("   Avvia con: streamlit run app.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
