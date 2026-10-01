from src.retrival.retriver import retriever
from src.generation.prompt import prompt
from src.generation.llm import llm
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from src.generation.question_rewriter import question_rewrite_prompt
from src.retrival.reranker import rerank_docs



def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )


def run_rag(inputs):
    question = inputs["question"]
    chat_history = inputs["chat_history"]
    # 1. Rewrite the question
    rewrite_messages = question_rewrite_prompt.invoke({
        "question": question,
        "chat_history": chat_history
    })

    standalone_question = llm.invoke(rewrite_messages)

    # 2. Use rewritten question for retrieval
    retrieved_docs = retriever.invoke(standalone_question)

    # --------------------------------
    # 3. Rerank retrieved documents
    # --------------------------------

    reranked_docs = rerank_docs(
        standalone_question,
        retrieved_docs,
        top_n=3
    )
    # --------------------------------
    # 4. Format reranked documents
    # --------------------------------

    context = format_docs(
        reranked_docs
    )

    # # 3. Create textbook context
    # context = format_docs(retrieved_docs)

    # 4. Use original question + context + history for final answer
    messages = prompt.invoke({
        "context": context,
        "question": question,
        "chat_history": chat_history
    })


    # 4. Generate answer
    answer = llm.invoke(messages)

    pages = set()

    for doc in reranked_docs:
        page = doc.metadata.get("page", 0) + 1
        pages.add(page)

    sources = []

    for page in sorted(pages):
        sources.append(f"Biology 2e, Page {page}")

    # 6. Return both answer and sources
    return {
        "answer": answer,
        "sources": sources
    }


rag_chain = RunnableLambda(run_rag)
chat_history = []

while True:

    question = input("\nAsk a question: ")

    if question.lower() in ["exit", "quit"]:
        print("\nGoodbye!")
        break

    response = rag_chain.invoke({
        "question": question,              #ekhane eta dictionary
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

# def format_docs(docs):
#     return "\n\n".join(
#         doc.page_content for doc in docs
#     )


# question = "Why do plants need sunlight?"


# # --------------------------------
# # 1. Retrieve documents
# # --------------------------------

# retrieved_docs = retriever.invoke(question)

# print("\n\n========== RETRIEVED DOCUMENTS ==========")

# for i, doc in enumerate(retrieved_docs):

#     print(f"\n--- Document {i + 1} ---")
#     print(doc.page_content[:500])

#     print("\nMetadata:")
#     print(doc.metadata)


# # --------------------------------
# # 2. Create context
# # --------------------------------

# context = format_docs(retrieved_docs)

# print("\n\n========== FORMATTED CONTEXT ==========")
# print(context[:2000])


# # --------------------------------
# # 3. Create prompt
# # --------------------------------

# messages = prompt.invoke({
#     "context": context,
#     "question": question
# })

# print("\n\n========== PROMPT ==========")
# print(messages)


# # --------------------------------
# # 4. Generate answer
# # --------------------------------

# response = llm.invoke(messages)

# print("\n\n========== FINAL ANSWER ==========")
# print(response)
