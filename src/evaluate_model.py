# evaluate_model.py

from rag_pipeline import load_rag_chain, ask
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, EMBEDDING_MODEL, MODEL_NAME
import pandas as pd

# ---------------------------
# 1. Define models
# ---------------------------
models_info = [
    {"name": "MiniLM", "embedding_model": "all-MiniLM-L6-v2"},
    {"name": "MPNet", "embedding_model": "all-mpnet-base-v2"}
]

# ---------------------------
# 2. Define evaluator LLM (uses cross-model evaluation)
# ---------------------------
def create_evaluator(model_name, embedding_model):
    # For simplicity, evaluator uses same ChatGroq interface, but different embedding/model
    return ChatGroq(
        model_name=model_name,
        temperature=0,
        groq_api_key=GROQ_API_KEY
    )

# ---------------------------
# 3. Evaluation prompt
# ---------------------------
evaluation_prompt = PromptTemplate(
    input_variables=["question", "answer", "context"],
    template="""
You are an expert evaluator.

Evaluate the answer based ONLY on the given context.

Criteria:
1. Correctness (1-5)
2. Relevance (1-5)
3. Groundedness (1-5)

Return ONLY this format:
Correctness: X
Relevance: X
Groundedness: X
Overall: X

Question: {question}
Context: {context}
Answer: {answer}
"""
)

# ---------------------------
# 4. Questions to evaluate
# ---------------------------
questions = [
    "who is the examiner for this course?",
    "What are the key topics covered in this course?",
    "what programming languages are used?",
    "what is module4 in this course",
    "What are the learning objectives?",
]

# ---------------------------
# 5. Generate answers from both models
# ---------------------------
answers = {}

for model in models_info:
    print(f"\nGenerating answers using {model['name']}...")
    # Load RAG chain with this embedding/model
    rag_chain, llm, prompt = load_rag_chain()
    
    model_answers = []
    for q in questions:
        answer, sources, context = ask(q, rag_chain, llm, prompt)
        model_answers.append({
            "question": q,
            "answer": answer,
            "sources": ", ".join(sources),
            "context": context
        })
    answers[model['name']] = model_answers

# ---------------------------
# 6. Cross-evaluation
# ---------------------------
evaluations = []

for q_idx, q in enumerate(questions):
    # MiniLM answer evaluated by MPNet
    mini_answer_info = answers["MiniLM"][q_idx]
    mpnet_evaluator = create_evaluator(MODEL_NAME, "all-mpnet-base-v2")
    eval_chain = evaluation_prompt | mpnet_evaluator | StrOutputParser()
    eval_mini_by_mpnet = eval_chain.invoke({
        "question": q,
        "answer": mini_answer_info["answer"],
        "context": mini_answer_info["context"]
    })

    # MPNet answer evaluated by MiniLM
    mpnet_answer_info = answers["MPNet"][q_idx]
    minilm_evaluator = create_evaluator(MODEL_NAME, "all-MiniLM-L6-v2")
    eval_chain2 = evaluation_prompt | minilm_evaluator | StrOutputParser()
    eval_mpnet_by_mini = eval_chain2.invoke({
        "question": q,
        "answer": mpnet_answer_info["answer"],
        "context": mpnet_answer_info["context"]
    })

    # Combine info
    evaluations.append({
        "question": q,
        "MiniLM Answer": mini_answer_info["answer"],
        "MiniLM Sources": mini_answer_info["sources"],
        "MiniLM→MPNet Eval": eval_mini_by_mpnet,
        "MPNet Answer": mpnet_answer_info["answer"],
        "MPNet Sources": mpnet_answer_info["sources"],
        "MPNet→MiniLM Eval": eval_mpnet_by_mini
    })

# ---------------------------
# 7. Save to CSV
# ---------------------------
df = pd.DataFrame(evaluations)
df.to_csv("cross_model_evaluation.csv", index=False)
print("\nCross-model evaluation complete! Saved as cross_model_evaluation.csv")