"""from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Embeddings loaded successfully")


from langchainchains import RetrievalQA
print("RetrievalQA loaded successfully")
"""
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('data/Syllabus_for_DAT410___DIT728_Design_of_AI_systems.pdf')
docs = loader.load()
print(f'Pages loaded: {len(docs)}')
for d in docs:
    print(f'Page {d.metadata[\"page\"]}: {len(d.page_content)} characters')
    print(f'Preview: {d.page_content[:100]}')
    print('---')