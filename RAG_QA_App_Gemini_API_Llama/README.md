# 📄 RAG-based PDF Question Answering (Gemini + LlamaIndex)

This project implements a **PDF Question Answering (QA) system** using:

- [LlamaIndex](https://docs.llamaindex.ai) for document indexing and retrieval  
- **Gemini (Google)** for embeddings + LLM  
- A simple pipeline to **upload PDFs, embed them, and query with natural language**

---

## 🚀 Features
- Load PDF documents and split them into chunks
- Create embeddings using **Gemini Embedding model**
- Store embeddings in a **Vector Index** (in-memory)
- Query using Gemini LLM with context from the PDF

---