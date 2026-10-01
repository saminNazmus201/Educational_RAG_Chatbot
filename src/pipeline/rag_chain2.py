from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough
)

from src.retrival.retriver import retriever
from src.generation.prompt import prompt
from src.generation.llm import llm


# --------------------------------
# 1. Format retrieved documents
# --------------------------------

def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )


# --------------------------------
# 2. Extract sources
# --------------------------------

def get_sources(docs):

    pages = set()

    for doc in docs:
        page = doc.metadata.get("page", 0) + 1
        pages.add(page)

    sources = []

    for page in sorted(pages):
        sources.append(f"Biology 2e, Page {page}")

    return sources


# --------------------------------
# 3. Retrieve documents ONCE
# --------------------------------

retrieve = retriever


# --------------------------------
# 4. Build RAG chain
# --------------------------------

rag_chain = (
    RunnableParallel(
        question=RunnablePassthrough(),
        chat_history=RunnableLambda(
            lambda x: x["chat_history"]
        ),
        documents=RunnableLambda(
            lambda x: retrieve.invoke(x["question"])
        )
    )
    | RunnableParallel(
        context=RunnableLambda(
            lambda x: format_docs(x["documents"])
        ),
        question=RunnableLambda(
            lambda x: x["question"]
        ),
        chat_history=RunnableLambda(
            lambda x: x["chat_history"]
        ),
        documents=RunnableLambda(
            lambda x: x["documents"]
        )
    )
    | RunnableParallel(
        answer=(
            {
                "context": RunnableLambda(
                    lambda x: x["context"]
                ),
                "question": RunnableLambda(
                    lambda x: x["question"]
                ),
                "chat_history": RunnableLambda(
                    lambda x: x["chat_history"]
                )
            }
            | prompt
            | llm
        ),
        documents=RunnableLambda(
            lambda x: x["documents"]
        )
    )
    | RunnableLambda(
        lambda x: {
            "answer": x["answer"],
            "sources": get_sources(x["documents"])
        }
    )
)


# --------------------------------
# 5. Conversation loop
# --------------------------------

chat_history = []

while True:

    question = input("\nAsk a question: ")

    if question.lower() in ["exit", "quit"]:
        print("\nGoodbye!")
        break

    response = rag_chain.invoke({
        "question": question,
        "chat_history": chat_history
    })

    print("\n========== ANSWER ==========")
    print(response["answer"])

    print("\n========== SOURCES ==========")

    for source in response["sources"]:
        print(f"- {source}")

    chat_history.append({
        "role": "user",
        "content": question
    })

    chat_history.append({
        "role": "assistant",
        "content": response["answer"]
    })