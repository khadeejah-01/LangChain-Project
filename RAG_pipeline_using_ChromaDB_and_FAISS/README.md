# RAG Pipeline using ChromaDB and FAISS

A small Retrieval-Augmented Generation (RAG) project exploring document embeddings, vector stores, similarity search, and retrieval using **LangChain**, **ChromaDB**, and **FAISS**.

## 1. Project Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```cmd
.venv\Scripts\activate.bat
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## 2. Embeddings

This project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

It produces **384-dimensional embeddings** and is lightweight enough for local/CPU-based use.

### SentenceTransformer vs HuggingFaceEmbeddings

`SentenceTransformer` is the direct way to generate embeddings:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embedding = model.encode("Some text")
```

Here, `.encode()` must be called manually.

For LangChain, we can use:

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

`HuggingFaceEmbeddings` acts as an **adapter/interface** that makes the SentenceTransformer model usable by LangChain components such as Chroma and FAISS.

```text
SentenceTransformer
       │
       │  encode()
       ▼
HuggingFaceEmbeddings
       │
       │  LangChain embedding interface
       ▼
 Chroma / FAISS
```

The same embedding model is used to embed both the stored documents and user queries.

---

## 3. ChromaDB

Chroma can automatically embed documents when an embedding model is provided:

```python
db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)
```

It can also be given plain text using:

```python
Chroma.from_texts(...)
```

### Persistent ChromaDB

A local vector store does not necessarily have to be persistent.

* **Local** → the vector store runs on your machine.
* **Persistent** → the vector-store data is saved and can be reused after the Python process ends.

For persistent storage:

```python
db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_langchain_db"
)
```

This is useful when documents should be embedded **once** and reused later instead of being embedded every time.

An existing persistent Chroma database can later be opened with:

```python
db = Chroma(
    collection_name="my_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)
```

---

## 4. Chroma Similarity Search

A query can be searched directly:

```python
query = "Who are the authors of Attention Is All You Need?"

retrieved_results = db.similarity_search(query)

print(retrieved_results[0].page_content)
```

The Chroma vector store uses the embedding function provided to it to convert the query into an embedding before performing the similarity search.

Conceptually:

```text
User Query
    ↓
Embedding Model
    ↓
Query Vector
    ↓
Chroma
    ↓
Similar Document Chunks
```

---

## 5. FAISS

FAISS is primarily a **vector indexing and similarity-search library**. In LangChain, the FAISS vector store adds document management around the underlying FAISS index.

```python
from langchain_community.vectorstores import FAISS

db = FAISS.from_documents(
    documents=docs,
    embedding=embeddings
)
```

Conceptually, the LangChain FAISS vector store contains:

```text
LangChain FAISS Vector Store
        │
   ┌────┼──────────────┐
   ↓    ↓              ↓
 FAISS  Docstore    ID Mapping
 Index
   │      │              │
vectors Documents   index → document ID
```

FAISS itself mainly answers:

> Which vectors are closest to this query vector?

For example, it may find vector positions:

```text
[7, 2, 11]
```

LangChain then maps those positions back to the corresponding documents.

Conceptually:

```text
FAISS position
      ↓
Document ID
      ↓
Document
```

LangChain maintains an internal mapping such as:

```text
FAISS index position    Document ID
        0       →       UUID_A
        1       →       UUID_B
        2       →       UUID_C
```

and a document store:

```text
UUID_A → Document 0
UUID_B → Document 1
UUID_C → Document 2
```

Thus, the **LangChain FAISS wrapper** manages the document IDs, document store, and mapping around the underlying FAISS index.

### FAISS Similarity Search

```python
query = "Who are the authors of Attention Is All You Need?"

retrieved_results = db.similarity_search(query)

print(retrieved_results[0].page_content)
```

The general flow is:

```text
Query
  ↓
Embedding Model
  ↓
Query Vector
  ↓
FAISS Similarity Search
  ↓
Nearest Vector Positions
  ↓
LangChain ID Mapping
  ↓
Documents
```

---

## 6. Hugging Face Hub

The `huggingface_hub` library provides the connection between the local environment and the Hugging Face Model Hub.

It handles:

* Downloading model files
* Downloading tokenizer/configuration files
* Local model caching
* Reusing already downloaded models

Therefore, after a model has been downloaded and cached locally, it usually does not need to be downloaded again.

---

## 7. RAG Pipeline

The overall pipeline explored in this project is:

```text
Documents
    ↓
Text Splitting
    ↓
Embeddings
    ↓
ChromaDB / FAISS
    ↓
Similarity Search / Retriever
    ↓
Relevant Document Chunks
    ↓
LLM
    ↓
Answer
```

The main idea is to **retrieve relevant information from the documents before generating an answer**.

---

## 8. LangChain Text Splitters

Text splitters are now provided through the separate package:

```bash
pip install langchain-text-splitters
```

and imported using:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

LangChain separated these components into their own package as the library became more modular.

---

## Technologies

* Python
* LangChain
* Sentence Transformers
* Hugging Face
* ChromaDB
* FAISS
* Vector Embeddings
* Retrieval-Augmented Generation (RAG)
