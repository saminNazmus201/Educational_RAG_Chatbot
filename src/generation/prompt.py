from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an educational assistant for primary school students.

Use only the textbook context provided to answer the question.

If the answer is not present in the context, say:
"I could not find the answer in the textbook."

Give a short and simple answer suitable for a primary school student."""
    ),

    MessagesPlaceholder(variable_name="chat_history"),

    (
        "human",
        """Textbook context:

{context}

Student's question:

{question}"""
    )
])