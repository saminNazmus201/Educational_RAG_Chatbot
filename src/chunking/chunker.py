from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


pdf_path = "data/raw/ConceptsOfBiology-LR.pdf"

loader = PyMuPDFLoader(pdf_path)

documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Original documents: {len(documents)}")
print(f"Total chunks: {len(chunks)}")

print("\n---chunk ---")
print(chunks[50].page_content)
print("\n---chunk metadata ---")
print(chunks[50].metadata)