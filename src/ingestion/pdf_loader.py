from langchain_community.document_loaders import PyMuPDFLoader


pdf_path = "data/raw/ConceptsOfBiology-LR.pdf"

loader = PyMuPDFLoader(pdf_path)

documents = loader.load()

print(f"Number of pages: {len(documents)}")

print("\n--- First page text ---")
print(documents[50].page_content)

print("\n--- First page metadata ---")
print(documents[50].metadata)