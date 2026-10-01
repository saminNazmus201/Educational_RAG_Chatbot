from langchain_core.prompts import ChatPromptTemplate


question_rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Rewrite the student's question into a standalone question.

Use the conversation history to understand words such as:
"it", "they", "this", "that", or "its".

Do not answer the question.
Only return the rewritten question."""
    ),
    (
        "human",
        """Conversation history:

{chat_history}

Student's question:

{question}"""
    )
])