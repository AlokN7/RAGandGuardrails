"""
RAG & Guardrails Evaluation: Master Knowledge Ingestion
"""

import os
import shutil
import glob
import pandas as pd
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 50

# Dynamically lock the base directory to exactly where this script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Absolute paths for Source folders
SOURCE_CLEAN_DIR = os.path.join(BASE_DIR, "data_clean")
SOURCE_POISONED_DIR = os.path.join(BASE_DIR, "data_poisoned")

# Absolute paths for Output folders
DB_CLEAN_OUT = os.path.join(BASE_DIR, "chroma_clean")
DB_POISONED_OUT = os.path.join(BASE_DIR, "chroma_poisoned")

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def load_clean_documents(source_folder):
    """Loads and splits the benign .txt files."""
    print(f"[*] Loading clean documents from: {source_folder}")
    loader = DirectoryLoader(
        source_folder, 
        glob="./*.txt", 
        loader_cls=TextLoader, 
        loader_kwargs={'encoding': 'utf-8'}
    )
    documents = loader.load()
    
    if not documents:
        print(f"[!] Warning: No .txt files found in {source_folder}.")
        return []

    print(f"[*] Found {len(documents)} clean .txt files. Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

def load_poisoned_documents(source_folder):
    """Loads the unbroken poisoned CSV rows as LangChain Documents."""
    print(f"[*] Scanning for poisoned CSV batches in: {source_folder}")
    
    # Matches poisoned_batch1.csv, poisoned_batch_1.csv, etc.
    csv_pattern = os.path.join(source_folder, "poisoned_batch*.csv")
    csv_files = glob.glob(csv_pattern)
    
    all_data = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            all_data.append(df)
            print(f"  [+] Loaded: {os.path.basename(file_path)} ({len(df)} rows)")
        except Exception as e:
            print(f"  [!] Error reading {file_path}: {e}")
            
    if not all_data:
        print(f"[!] Warning: No CSV files matching 'poisoned_batch*.csv' found in {source_folder}.")
        return []
        
    master_df = pd.concat(all_data, ignore_index=True)
    
    print(f"[*] Converting {len(master_df)} poisoned rows into RAG documents...")
    documents = []
    for index, row in master_df.iterrows():
        # Ensure we only process rows that actually have data
        if pd.notna(row.get('poisoned_context')):
            doc = Document(
                page_content=str(row['poisoned_context']),
                metadata={
                    "source": f"advbench_query_{index}", 
                    "goal": str(row.get('goal', 'unknown'))
                }
            )
            documents.append(doc)
            
    return documents

def build_db(db_path, documents):
    """Builds or rebuilds a ChromaDB vector store from a list of documents."""
    if os.path.exists(db_path):
        print(f"[*] Purging existing vector store: {db_path}")
        shutil.rmtree(db_path)
        
    print(f"[*] Building Vector Index at {db_path} with {len(documents)} total chunks/documents...")
    Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory=db_path
    )
    print(f"[+] Success: {db_path} fully built.\n")

# --- EXECUTION BLOCK (This was missing!) ---
if __name__ == "__main__":
    print("--- Initialize RAG Poisoning Environment ---\n")
    
    # 1. Process the clean background data
    clean_docs = load_clean_documents(SOURCE_CLEAN_DIR)
    
    # 2. Build the baseline database (Clean Only)
    print("\n--- Creating Baseline Database (chroma_clean) ---")
    if clean_docs:
        build_db(DB_CLEAN_OUT, clean_docs)
    else:
        print("[!] Skipping chroma_clean creation due to missing source files.")
    
    # 3. Process the malicious payloads
    print("\n--- Loading Poisoned Payloads ---")
    poisoned_docs = load_poisoned_documents(SOURCE_POISONED_DIR)
    
    # 4. Build the experimental database (Clean + Poisoned)
    print("\n--- Creating Experimental Database (chroma_poisoned) ---")
    combined_docs = clean_docs + poisoned_docs
    if combined_docs:
        build_db(DB_POISONED_OUT, combined_docs)
    else:
        print("[!] Skipping chroma_poisoned creation due to missing source files.")
        
    print("\n✅ Ingestion Complete. Vector stores are primed for evaluation.")