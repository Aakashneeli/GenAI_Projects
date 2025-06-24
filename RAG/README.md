# Explorations in Retrieval-Augmented Generation (RAG)

This repository contains a collection of my personal projects and learning notebooks focused on Retrieval-Augmented Generation (RAG). The projects range from foundational, step-by-step notebooks to a fully interactive web application, demonstrating various techniques for enhancing Large Language Models (LLMs) with external knowledge.

## Projects Overview

This folder contains several distinct RAG implementations:

### 1. LawBot: AI Legal Assistant (`LawBot.py`)

A fully-functional Streamlit web application that acts as an AI legal assistant. It answers questions based on the content of the Constitution of India (`COI.pdf`).

***Core Functionality:** Users can ask questions in a chat interface, and the application uses a RAG pipeline to retrieve relevant sections from the PDF and generate a context-aware answer.
* **Technology:** Built with Streamlit, LangChain, and LangGraph. [cite_start]It uses a FAISS vector store and OpenAI models (`gpt-4o-mini`, `text-embedding-3-large`).
* **Key Feature:** Demonstrates a complete, deployable RAG application.

### 2. Conversational RAG Agent with Memory (`rag_app_memory.py`, `rag_app_memory.ipynb`)

This project implements a more advanced, stateful RAG agent using LangGraph. It maintains conversation history, allowing for follow-up questions.

* [cite_start]**Core Functionality:** The agent can decide whether it needs to retrieve information from the source document (`COI.pdf`) to answer a question or if it can respond based on the conversation history alone.
* [cite_start]**Technology:** Uses LangGraph's `MessagesState` to manage memory, a tool-calling router to decide when to retrieve, and FAISS for the vector store.
* **Key Feature:** Showcases how to build conversational memory and agentic logic into a RAG system.

### 3. Foundational RAG Pipeline (`rag_indexing.ipynb`)

A Jupyter notebook that provides a clear, step-by-step walkthrough of a basic RAG pipeline.

* [cite_start]**Core Functionality:** Loads a PDF (`rag.pdf`), splits it into chunks, generates embeddings, indexes them in FAISS, and performs a similarity search to answer a query.
* **Key Feature:** An excellent educational resource for understanding the fundamental "Indexing, Retrieval, Generation" workflow.

### 4. RAG with Web Content (`rag_webpages.ipynb`)

This notebook demonstrates how to apply the RAG technique to live web content.

* [cite_start]**Core Functionality:** It uses `WebBaseLoader` to scrape content from a specified blog post, then indexes and queries the content using a simple LangGraph chain.
* **Key Feature:** Illustrates the flexibility of RAG by using a webpage as a dynamic knowledge source.

---

## Technologies Used

* [cite_start]**Frameworks:** LangChain, LangGraph, Streamlit 
* [cite_start]**LLMs & Embeddings:** OpenAI (`gpt-4o-mini`, `gpt-3.5-turbo-instruct`, `text-embedding-3-large`) 
* [cite_start]**Vector Store:** FAISS (via `faiss-cpu`) 
* [cite_start]**Document Loading:** PyPDF, WebBaseLoader 
* [cite_start]**Environment Management:** python-dotenv 

---

## Setup and Installation

Follow these steps to set up the environment and run the projects.

**1. Clone the Repository**
```bash
git clone <your-repository-url>
cd RAG

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# .env file
OPENAI_API_KEY="your-openai-api-key-here"
# Optional: For LangSmith tracing
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="your-langsmith-api-key-here"


