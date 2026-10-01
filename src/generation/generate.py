from prompt import prompt
from llm import llm


context = """
Plants use sunlight to make food through a process called photosynthesis.
Sunlight provides the energy needed for this process.
"""

question = "Why do plants need sunlight?"


messages = prompt.invoke({
    "context": context,
    "question": question
})


response = llm.invoke(messages)


print("\nANSWER:")
print(response)