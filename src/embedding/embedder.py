from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# --------------------------------
# 1. Load PDF
# --------------------------------

pdf_path = "data/raw/ConceptsOfBiology-LR.pdf"

loader = PyMuPDFLoader(pdf_path)

documents = loader.load()


# --------------------------------
# 2. Split documents into chunks
# --------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)


# --------------------------------
# 3. Create embedding model
# --------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

query = "Why do plants need sunlight?"


texts= [chunk.page_content for chunk in chunks]
document_vector= embedding_model.embed_documents(texts)
query_vector= embedding_model.embed_query(query)

print("Number of documents:", len(documents))
print("Number of chunks:", len(chunks))
print("Number of documents vector:", len(document_vector))
print("Dimensions of first vector:", len(document_vector[0]))
print(
    "Query vector dimensions:",
    len(query_vector)
)

print("First 10 values:")
print(document_vector[0][:10])