from sentence_transformers import CrossEncoder


# --------------------------------
# 1. Load reranking model
# --------------------------------

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# --------------------------------
# 2. Rerank retrieved documents
# --------------------------------

def rerank_docs(question, docs, top_n=3):

    # Create question-document pairs
    pairs = []

    for doc in docs:
        pairs.append(
            (question, doc.page_content)
        )

    # Get relevance scores
    scores = reranker.predict(pairs)

    # Combine documents with scores
    results = list(
        zip(docs, scores)
    )

    # Sort from highest score to lowest
    results = sorted(
        results,
        key=lambda x: x[1],
        reverse=True
    )

    # Keep only the top N documents
    reranked_docs = []

    for doc, score in results[:top_n]:
        reranked_docs.append(doc)

    return reranked_docs