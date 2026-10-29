# 📚 Educational RAG Chatbot

**An AI-powered educational assistant that helps students understand textbook content through Retrieval-Augmented Generation (RAG).**

The Educational RAG Chatbot is a project focused on building a question-answering assistant that retrieves relevant information from educational textbooks and uses a language model to generate context-aware answers.

The goal is to make learning easier by allowing students to ask questions in natural language and receive answers grounded in their study materials.

> 🚧 **Project Status:** In Development  
> This project is being developed incrementally. More features and improvements will be added as development progresses.

---

## ✨ Project Goals

- 📖 **Textbook-Based Question Answering** — Answer questions using information retrieved from educational materials.
- 🔍 **Semantic Retrieval** — Find relevant textbook passages using vector embeddings.
- 🧠 **Context-Aware Generation** — Generate answers based on retrieved context rather than relying solely on the language model's internal knowledge.
- 📑 **Source References** — Track the source pages associated with retrieved textbook content.
- 💬 **Interactive Chat Interface** — Build a user-friendly interface for asking questions and exploring textbook content.

---

## 🏗️ System Architecture

The project follows a Retrieval-Augmented Generation (RAG) pipeline.

```text
             Educational Textbook (PDF)
                        |
                        ▼
                 PDF Text Extraction
                        |
                        ▼
                  Text Chunking
                        |
                        ▼
                Text Embeddings
                        |
                        ▼
                 Chroma Vector DB
                        |
                        ▼
                  User Question
                        |
                        ▼
               Question Rewriting
                        |
                        ▼
                Relevant Retrieval
                        |
                        ▼
                  Reranking
                        |
                        ▼
               Context Construction
                        |
                        ▼
               Prompt Construction
                        |
                        ▼
                Qwen Language Model
                        |
                        ▼
                Generated Answer
                        |
                        ▼
               Answer + Source Pages
```

*Note: This diagram represents the intended/currently developed pipeline. The complete interactive application and document-upload workflow are still under development.*

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangChain | Building and connecting components of the RAG pipeline |
| PyMuPDF | Extracting text and metadata from PDF documents |
| RecursiveCharacterTextSplitter | Splitting textbook content into manageable chunks |
| Hugging Face Transformers | Loading and running language models |
| Sentence Transformers | Generating text embeddings |
| ChromaDB | Storing and retrieving vector embeddings |
| Qwen2.5-0.5B-Instruct | Language model for answer generation |
| Cross-Encoder | Reranking retrieved passages |
| Streamlit | Developing the interactive web interface |

---

## 🔄 How It Works

### 1. PDF Text Extraction
Textbook content is extracted from PDF documents using PyMuPDF.

### 2. Text Chunking
The extracted text is divided into smaller, overlapping chunks to make retrieval more effective.

Current chunking configuration:

- **Chunk size:** 1,000 characters
- **Chunk overlap:** 200 characters

### 3. Text Embeddings
Each text chunk is converted into a numerical vector using:

`sentence-transformers/all-MiniLM-L6-v2`

The embedding model produces **384-dimensional vectors**.

### 4. Vector Storage and Retrieval
The generated embeddings are stored in ChromaDB. When a question is submitted, the system retrieves potentially relevant textbook passages.

### 5. Question Rewriting
A question-rewriting component helps reformulate user questions to improve retrieval.

### 6. Passage Reranking
A cross-encoder reranks the retrieved passages to prioritize the most relevant information.

Current reranker:

`cross-encoder/ms-marco-MiniLM-L-6-v2`

### 7. Answer Generation
The retrieved passages are provided as context to the Qwen language model, which generates an answer based on the available textbook information.

### 8. Source References
The pipeline tracks source-page metadata so that relevant textbook pages can be returned alongside the generated answer.

---

## 🤖 Models

### Embedding Model
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Embedding dimension:** 384

### Language Model
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct`

### Reranking Model
- **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`

---

## 📂 Project Structure

```text
Educational-RAG/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── chunking/
│   │   └── chunker.py
│   │
│   ├── generation/
│   │   └── question_rewriter.py
│   │
│   ├── retrieval/
│   │   └── ...
│   │
│   └── ...
│
├── .gitignore
├── requirements.txt
└── README.md
```

*The directory tree is illustrative. Update it to match the actual repository structure as the project evolves.*

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.10 or later is recommended.
- pip
- Git

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Educational-RAG
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with your repository's Git URL.

### 2. Create a Virtual Environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Chunking Module

From the project root directory:

```bash
python -m src.chunking.chunker
```

This runs the PDF text extraction and chunking module using the configured textbook PDF.

**Note:** Ensure that the required PDF exists at the path configured in the chunking script before running it.

### 5. Launch the Streamlit Interface

Once the Streamlit application is configured:

```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Current Development Progress

| Component | Status |
|---|---|
| PDF text extraction | Implemented |
| Text chunking | Implemented |
| Chunk token-count analysis | Implemented |
| Hugging Face embedding model | Integrated into the pipeline |
| ChromaDB vector storage | Under development / integration |
| Question rewriting | Developed |
| Passage retrieval | Under development / integration |
| Cross-encoder reranking | Developed |
| Context-aware answer generation | Under development / integration |
| Streamlit chat interface | In development |
| PDF upload through the interface | Planned |
| Persistent chat history | Planned |

*Development status will be updated as the project progresses.*

---

## 🚀 Future Improvements

Planned improvements include:

- [ ] Complete the end-to-end RAG pipeline.
- [ ] Connect the Streamlit interface to the RAG backend.
- [ ] Allow users to upload their own textbook PDFs.
- [ ] Build a dynamic document-ingestion workflow.
- [ ] Display source pages alongside answers.
- [ ] Add recent conversation history.
- [ ] Improve retrieval quality and answer grounding.
- [ ] Evaluate answer accuracy and retrieval performance.
- [ ] Improve error handling and user experience.

---

## 🎯 Motivation

Students often need to search through lengthy textbooks to find explanations for specific concepts. This project explores how Retrieval-Augmented Generation can make educational content more accessible by combining semantic search with language-model-based answer generation.

The long-term objective is to develop a practical educational assistant that helps students interact with their learning materials more efficiently.

---

## 👩‍💻 Author

**Tanjila Ahmed Medha**

MSc Student — AI and Language

Stockholm University, Sweden

---

⭐ *This project is a work in progress. Feedback, suggestions, and contributions are welcome.*
