# 📚 Groq RAG Chatbot using LangChain, FAISS & Streamlit

A Retrieval-Augmented Generation (RAG) chatbot that allows users to ask questions from their own PDF documents. The application processes PDFs into vector embeddings using **Ollama Embeddings**, stores them in a **FAISS Vector Database**, retrieves relevant document chunks, and generates context-aware responses using **Groq's Llama 3.1 8B Instant** model.

---

## 🚀 Features

- 📄 Load multiple PDF documents
- ✂️ Automatic document chunking
- 🧠 Generate vector embeddings using Ollama
- 💾 Store embeddings in FAISS Vector Store
- 🔍 Semantic document retrieval
- 🤖 Groq Llama 3.1 8B Instant for answer generation
- 💬 Interactive Streamlit Web UI
- 📚 View retrieved document context
- ⚡ Fast and lightweight RAG pipeline

---

## 🏗️ Project Architecture

```
PDF Documents
      │
      ▼
PyPDFDirectoryLoader
      │
      ▼
RecursiveCharacterTextSplitter
      │
      ▼
Ollama Embeddings
      │
      ▼
FAISS Vector Store
      │
      ▼
Retriever
      │
      ▼
Groq Llama 3.1 8B Instant
      │
      ▼
Answer Generation
      │
      ▼
Streamlit Interface
```

---

## 📂 Project Structure

```
Groq-RAG-Chatbot/
│
├── data/
│   ├── sample.pdf
│
├── groq_rag.py
├── requirements.txt
├── .env
├── README.md

```

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Frameworks
- Streamlit
- LangChain

### LLM
- Groq
- Llama 3.1 8B Instant

### Embedding Model
- Ollama
- nomic-embed-text

### Vector Database
- FAISS

### Document Loader
- PyPDF

### Environment Variables
- python-dotenv

---

## 📦 Python Libraries Used

```
streamlit
langchain
langchain-community
langchain-core
langchain-classic
langchain-groq
langchain-ollama
langchain-text-splitters
faiss-cpu
python-dotenv
pypdf
ollama
```



### Create Virtual Environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Install Ollama

Download from

https://ollama.com/download

---

## Pull Embedding Model

```bash
ollama pull nomic-embed-text
```

---

## Start Ollama Server

```bash
ollama serve
```

---

## Create .env File

```env
GROQ_API_KEY=your_groq_api_key

LANGCHAIN_API_KEY=your_langsmith_api_key

LANGCHAIN_TRACING_V2=true

LANGCHAIN_PROJECT=Groq RAG Chatbot
```

---

## ▶️ Run Application

```bash
streamlit run groq_rag.py
```

---

## 💻 How It Works

### Step 1

Click

```
Generate Embeddings
```

This will

- Load PDFs
- Split into chunks
- Generate embeddings
- Store vectors inside FAISS

---

### Step 2

Enter your question

Example

```
What is Machine Learning?
```

---

### Step 3

Click

```
Answer
```

The chatbot will

- Retrieve relevant chunks
- Send context to Groq LLM
- Generate an accurate response
- Display retrieved document context

---

## 📸 Screenshots

### Home Page

```
(Add Screenshot Here)
```

---

### Generated Answer

```
(Add Screenshot Here)
```

---

### Document Context

```
(Add Screenshot Here)
```

---


## 📈 Learning Outcomes

This project demonstrates

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Document Chunking
- Embedding Generation
- Vector Databases
- Prompt Engineering
- Large Language Models
- LangChain Pipelines
- Streamlit Deployment

---


## ⭐ If you found this project useful

Please consider giving it a ⭐ on GitHub!

---

## 📜 License

This project is licensed under the Apache License.