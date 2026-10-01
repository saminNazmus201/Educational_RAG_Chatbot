from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# retriever = vectorstore.as_retriever(
#     search_kwargs={"k": 8}
# )
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 8,
        "fetch_k": 20,
        "lambda_mult": 0.5
    }
)

# question = "Why do plants need sunlight?"

# # retrieved_docs= retriever.invoke(question)


# results = vectorstore.similarity_search_with_score(
#      question,
#      k=5
#  )

# print(f"\nQuestion: {question}")

# print(f"\nNumber of retrieved documents: {len(retrieved_docs)}")


# for i, doc in enumerate(retrieved_docs):

#     print(f"\n==============================")
#     print(f"RESULT {i + 1}")
#     print(f"==============================")

#     print("\nContent:")
#     print(doc.page_content)

#     print("\nMetadata:")
#     print(doc.metadata)

# for i, (doc, score) in enumerate(results):

#     print("\n==============================")
#     print(f"RESULT {i + 1}")
#     print("==============================")

#     print("\nScore:")
#     print(score)

#     print("\nPage:")
#     print(doc.metadata.get("page", "unknown"))

#     print("\nContent:")
#     print(doc.page_content[:500])

# question = "What is a cell?"

# for k in [3, 5, 8]:

#     print("\n\n################################")
#     print(f"          k = {k}")
#     print("################################")

#     results = vectorstore.similarity_search_with_score(
#         question,
#         k=k
#     )

#     for i, (doc, score) in enumerate(results):

#         print("\n------------------------------")
#         print(f"RESULT {i + 1}")
#         print("------------------------------")

#         print("Score:", score)
#         print("Page:", doc.metadata.get("page", "unknown"))
#         print("Content:")
#         print(doc.page_content[:300])