from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer


# ==================================================
# 1. LOAD PDF
# ==================================================

pdf_path = "data/raw/ConceptsOfBiology-LR.pdf"

loader = PyMuPDFLoader(pdf_path)

documents = loader.load()


# ==================================================
# 2. SPLIT DOCUMENTS INTO CHUNKS
# ==================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)


# ==================================================
# 3. LOAD TOKENIZER
# ==================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_TOKEN_LENGTH = 256

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ==================================================
# 4. COUNT TOKENS IN EVERY CHUNK
# ==================================================

token_counts = []

for chunk in chunks:

    tokens = tokenizer(
        chunk.page_content,
        add_special_tokens=True,
        truncation=False
    )["input_ids"]

    token_counts.append(len(tokens))


# ==================================================
# 5. CALCULATE TOKEN STATISTICS
# ==================================================

total_chunks = len(chunks)

over_limit = sum(
    count > MAX_TOKEN_LENGTH
    for count in token_counts
)

within_limit = total_chunks - over_limit

average_tokens = (
    sum(token_counts) / total_chunks
    if total_chunks > 0
    else 0
)

max_tokens = max(token_counts) if token_counts else 0

min_tokens = min(token_counts) if token_counts else 0


# ==================================================
# 6. PRINT CHUNKING SUMMARY
# ==================================================

print("\n========== CHUNKING SUMMARY ==========")

print(f"Original documents: {len(documents)}")
print(f"Total chunks: {total_chunks}")

print("\n========== TOKEN ANALYSIS ==========")

print(f"Minimum tokens per chunk: {min_tokens}")
print(f"Maximum tokens per chunk: {max_tokens}")
print(f"Average tokens per chunk: {average_tokens:.2f}")

print(f"\nConfigured model sequence length: {MAX_TOKEN_LENGTH}")
print(f"Chunks within limit: {within_limit}")
print(f"Chunks exceeding limit: {over_limit}")

if total_chunks > 0:

    percentage = (over_limit / total_chunks) * 100

    print(f"Percentage exceeding limit: {percentage:.2f}%")


# ==================================================
# 7. INSPECT THE FIRST 10 CHUNKS
# ==================================================

print("\n========== FIRST 10 CHUNKS ==========")

for i, chunk in enumerate(chunks[:10]):

    token_count = token_counts[i]

    status = (
        "EXCEEDS LIMIT"
        if token_count > MAX_TOKEN_LENGTH
        else "WITHIN LIMIT"
    )

    print(f"\n--- Chunk {i + 1} ---")
    print(f"Token count: {token_count}")
    print(f"Status: {status}")
    print(f"Text preview: {chunk.page_content[:150]}")


# ==================================================
# 8. SHOW CHUNKS EXCEEDING THE TOKEN LIMIT
# ==================================================

print("\n========== CHUNKS EXCEEDING LIMIT ==========")

exceeding_chunks = [
    (i, count)
    for i, count in enumerate(token_counts)
    if count > MAX_TOKEN_LENGTH
]

if exceeding_chunks:

    for i, count in exceeding_chunks[:10]:

        print(f"Chunk {i + 1}: {count} tokens")

    if len(exceeding_chunks) > 10:

        print(
            f"...and {len(exceeding_chunks) - 10} "
            "more chunks exceeding the limit."
        )

else:

    print("No chunks exceed the configured token limit.")


# ==================================================
# 9. INSPECT CHUNK 50
# ==================================================

if len(chunks) > 50:

    print("\n========== CHUNK 50 DETAILS ==========")

    print("\n--- Chunk 50 content ---")
    print(chunks[50].page_content)

    print("\n--- Chunk 50 token count ---")
    print(token_counts[50])

    print("\n--- Chunk 50 metadata ---")
    print(chunks[50].metadata)

else:

    print("\nThere are fewer than 51 chunks.")


# ==================================================
# 10. TEST TOKENIZER TRUNCATION
# ==================================================

print("\n========== TRUNCATION TEST ==========")

long_chunk_index = next(
    (
        i
        for i, count in enumerate(token_counts)
        if count > MAX_TOKEN_LENGTH
    ),
    None
)

if long_chunk_index is not None:

    text = chunks[long_chunk_index].page_content

    # Count all tokens before truncation
    original_tokens = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False
    )["input_ids"]

    # Count tokens after explicitly requesting truncation
    truncated_tokens = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_TOKEN_LENGTH
    )["input_ids"]

    print(f"Chunk number: {long_chunk_index + 1}")

    print(
        f"Original token count: {len(original_tokens)}"
    )

    print(
        f"Token count after truncation: {len(truncated_tokens)}"
    )

    print(
        f"Tokens removed by truncation: "
        f"{len(original_tokens) - len(truncated_tokens)}"
    )

    print(
        f"Original text length: {len(text)} characters"
    )

else:

    print("No chunks exceed the configured token limit.")
