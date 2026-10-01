from transformers import pipeline
from langchain_core.runnables import RunnableLambda


generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens=100,
    temperature=0.1,
    return_full_text=False
)


def generate_answer(prompt_value):

    messages = []

    for message in prompt_value.messages:

        if message.type == "system":
            role = "system"

        elif message.type == "human":
            role = "user"

        else:
            role = "assistant"

        messages.append({
            "role": role,
            "content": message.content
        })

    result = generator(messages)

    return result[0]["generated_text"]


llm = RunnableLambda(generate_answer)


# text_generation_pipeline = pipeline(
#     "text-generation",
#     model="Qwen/Qwen2.5-0.5B-Instruct",
#     max_new_tokens=200,
#     temperature=0.3,
# )


# huggingface_llm = HuggingFacePipeline(
#     pipeline=text_generation_pipeline
# )


# llm = ChatHuggingFace(
#     llm=huggingface_llm
