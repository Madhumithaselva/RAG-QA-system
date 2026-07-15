**RAG COurse QA system**
A Retrival-Augmented Generation(RAG) system that answers questions about a university course,grounded strictly in official course materials (syllabus, lecture slides, instructions) to reduce hallucination compared to a standard LLM.

**Project Overview**
Traditional LLMs can generate answers but may hallucinate when they do not have access to specific domain knowledge.
This project implements a RAG pipeline where:
  1. Documents are extracted from PDFs.
  2. Text is split into smaller chunks.
  3. Chunks are converted into vector embeddings.
  4. Embeddings are stored in ChromaDB.
  5. User queries retrieve relevant document sections.
  6. Retrieved context is provided to the LLM to generate grounded answers.

**Architecture**
PDF documents
      |
PDF Text Extraction
      |
Text Chunking
      |
Embedding Generation
      |
ChromaDB Vector Store
      |
Similariry Search
      |
Relevant Context
      |
Groq Llama LLM
      |
Generated Answer

**Project Structure**
RAG-mini-project/

│
├── data/
│   └── Sample PDF documents
│
├── src/
│   ├── config.py
│   │       Configuration parameters
│   │
│   ├── embed_store.py
│   │       Document embedding and vector store creation
│   │
│   └── rag_pipeline.py
│           Retrieval-Augmented Generation pipeline
│
├── vectorstore/
│       ChromaDB vector database
│
├── RAG-mini-project.ipynb
│       Experimentation notebook
│
├── requirements.txt
│
├── testfile.py
│
└── README.md

**Technologies used**
-Python
-LangCHain
-ChromaDB
-Groq API
-Llama 3 LLM
-HuggingFace Sentence Transformers
-PyPDF

**Installation**
Clone the repository:
- git clone <repo url>
- cd RAG-mini-project
- Install required packages:
    bash
  pip install -r requirements.txt

  **Environment configuration**
  Create a '.env' file in the project root:
    GROQ_API_KEY=<api_key>

  The api key is loaded through the configuration module.

  **Running the project:**
  1. Add PDF documents inside the `data/` folder.
  2. Generate embeddings and create the vector store.
  3. Run the RAG pipeline to ask questions from the documents.

 **Example use case**
 The system can answer questions such as:
  - Summarize the uploaded document.
  - Explain a specific concept from the document.
  - Find information from course materials.
  - Answer questions basaed on provided PDFs.
    
**RAG pipeline COmponents**
   # Document Processing
    - Extracts text from PDF documents
    - Splits large documents into manageable chunks.
  # Embedding Generation
    - Converts text chunks into numerical vector representations using embedding models.
  # Vector Database
    - Stores document embeddings in ChromaDB.
    - Performs similarity-based retrieval.
  # Generation
    - Uses retrieved context with Groq Llama to generate answers.
    
**Current Limitations**
- No web interface currently available.
- Runs through Python scripts/notebook.
- Single document collection.
- No conversation memory.

**Future Improvements**
Planned improvements:
- FastAPI backend
- Streamlit user interface
- Better retrieval strategies
- Document metadata filtering
- Automated evaluation
- Docker deployment
