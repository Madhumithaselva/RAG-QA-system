from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import sys
import os
import shutil

import sys
print(sys.executable)

# Add project root to path so config can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import CHUNK_SIZE, CHUNK_OVERLAP, DATA_FOLDER, VECTORSTORE_FOLDER, EMBEDDING_MODEL


def create_vectorstore():

    # Step 1 — Load PDFs
    print("-----------------------------")
    print("---Loading PDFs...---")
    print("-----------------------------")

    if not os.path.exists(DATA_FOLDER):
        print(f"Data folder not found: {DATA_FOLDER}")
        return

    loader = PyPDFDirectoryLoader(DATA_FOLDER)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages")

    if len(documents) == 0:
        print("No pages loaded! Make sure PDFs exist in the data/ folder.")
        return

    # Remove empty pages
    documents = [doc for doc in documents if doc.page_content.strip()]
    print(f"{len(documents)} pages with actual text")

    if len(documents) == 0:
        print("All pages empty. Your PDFs may be scanned images.")
        return

    # Step 2 — Split into chunks
    print("-----------------------------")
    print(" Splitting documents...")
    print("-----------------------------")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    chunks = [c for c in chunks if c.page_content.strip()]

    print(f"Created {len(chunks)} chunks")

    if len(chunks) == 0:
        print("No chunks created.")
        return

    print("\nSample chunk:\n")
    print(chunks[0].page_content[:300])
    print("-----")

    # Step 3 — Create embeddings
    print("-----------------------------")
    print("\n🧠 Loading embedding model...")
    print("-----------------------------")

    embeddings = HuggingFaceEmbeddings(
        model_name=f"sentence-transformers/{EMBEDDING_MODEL}"
    )

    # Remove old vectorstore
    if os.path.exists(VECTORSTORE_FOLDER):
        shutil.rmtree(VECTORSTORE_FOLDER)
        print("🗑 Old vectorstore removed")

    # Step 4 — Create vector database
    print("-----------------------------")
    print("Creating vector database...")
    print("-----------------------------")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_FOLDER
    )

    # Save database
    vectorstore.persist()

    print("-----------------------------")
    print(f" Vectorstore created successfully")
    print("-----------------------------")
    print(f"Stored {len(chunks)} chunks")

    return vectorstore


if __name__ == "__main__":
    create_vectorstore()