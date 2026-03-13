from dotenv import load_dotenv
import os

import sys
print(sys.executable)
load_dotenv()

# API Key — loaded from .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Settings — change these for experiments
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
TOP_K = 4
DATA_FOLDER = "data/"
VECTORSTORE_FOLDER = "vectorstore/"
MODEL_NAME = "llama3-8b-8192"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Debug check

print("DATA_FOLDER:",DATA_FOLDER)
print("VECTORSTORE_FOLDER:",VECTORSTORE_FOLDER)
print("MODEL_NAME:",MODEL_NAME)
print("EMBEDDING_MODEL:",EMBEDDING_MODEL)

if GROQ_API_KEY:
    print("Config loaded successfully. API key found.")
else:
    print("API key not found. Check your .env file.")

