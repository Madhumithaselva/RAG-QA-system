from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import GROQ_API_KEY, TOP_K, VECTORSTORE_FOLDER, EMBEDDING_MODEL, MODEL_NAME

def load_rag_chain():
    # Load vectorstore
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_FOLDER,
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # Prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant for DAT410/DIT728.
Answer ONLY based on the context below.
If the answer is not in the context, say:
"I don't know based on the course materials."

Context: {context}
Question: {question}
Answer:"""
    )

    # LLM
    llm = ChatGroq(
        model_name=MODEL_NAME,
        temperature=0,
        groq_api_key=GROQ_API_KEY
    )

    # Format docs function
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Chain that ALSO returns source documents
    #from langchain_core.runnables import RunnableParallel





    rag_chain_with_sources = RunnableParallel(
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "source_documents": retriever
        }
    )

    return rag_chain_with_sources, llm, prompt

def ask(question, rag_chain_with_sources, llm, prompt):
    # Get answer and sources together
    result = rag_chain_with_sources.invoke(question)

    #Generate answer unisng only the retived context
    answer_chain = prompt | llm | StrOutputParser()
    answer = answer_chain.invoke({
        "context": result["context"],
        "question": question
    })
    
    # Get source documents
    docs = result["source_documents"]   
    
    # Show only unique sources with page numbers
    seen = set()
    sources = []
    for doc in docs:
        source = doc.metadata.get("source", "?").split("\\")[-1]
        page = doc.metadata.get("page", "?")
        key = f"{source} (page {page})"
        if key not in seen:
            seen.add(key)
            sources.append(key)

    return answer, sources

#Main Program
if __name__ == "__main__":
    print("Loading RAG chain...")
    rag_chain, llm, prompt = load_rag_chain()
    print("✅ Ready!")
    print("Ask a question: ")
    print("Type 'exit' to quit.\n")
    print("-" * 50)
    
    while True:
        question = input("Ask: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        print("\n⏳ Searching course materials...")
        answer, sources = ask(question, rag_chain, llm, prompt)

        print(f"\n🤖 Answer:\n{answer}")
        print(f"\n📚 Sources:\n{', '.join(sources)}")
        print("\n" + "-" * 50) 
        
        while True:
            # Get question from user
            question = input("\n❓ Your question: ").strip()

            # Exit condition
            if question.lower() in ["exit", "quit", "q"]:
                print("👋 Goodbye!")
                break

            # Skip empty input
            if not question:
                print("⚠️  Please type a question!")
                continue

            # Get answer
            print("\n⏳ Searching course materials...")
            answer, sources = ask(question, rag_chain, llm, prompt)

            # Show answer
            print("\n💬 Answer:")
            print(answer)

            # Show sources
            print("\n📄 Sources:")
            for s in sources:
                print(f"   - {s}")
            print("-" * 50)