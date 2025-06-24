# LawBot: AI Assistant for the Indian Constitution

LawBot is an AI-powered chatbot designed to answer questions about the Constitution of India. It leverages a Retrieval-Augmented Generation (RAG) pipeline to provide accurate and contextually relevant information from the constitutional text. The application features an interactive user interface built with Streamlit.

## Architecture

The project is divided into two main components:

1.  **`main.py` (Backend Logic):** This script contains the core `LawBot` class which encapsulates all the backend functionality. It is responsible for:
    * Loading and processing the source document (`COI.pdf`).
    * Splitting the document into manageable chunks.
    * Generating embeddings using Google's `embedding-001` model.
    * Creating and managing a FAISS vector store for efficient retrieval.
    * Setting up the RAG chain with a Google Gemini Pro model (`gemini-2.0-flash`), a prompt template, and a retriever.

2.  **`frontend.py` (User Interface):** This script builds a user-friendly, interactive web interface using Streamlit. Its responsibilities include:
    * Creating the page layout, title, and custom styling for a polished look and feel.
    * Managing the chat history and displaying the conversation between the user and the bot.
    * Handling user input and calling the backend `LawBot` instance to generate responses.
    * Displaying progress indicators and handling potential errors gracefully.

## Features

* **RAG-Powered Q&A:** Answers questions using information retrieved directly from the Constitution of India PDF.
* **Interactive Chat Interface:** A clean and modern UI built with Streamlit allows for an intuitive user experience.
* **Google Gemini Integration:** Utilizes the powerful and efficient `gemini-2.0-flash` model for response generation and `embedding-001` for creating text embeddings.
* **Efficient Retrieval:** Employs a FAISS vector store for fast and effective similarity searches.
* **Optimized Performance:** The backend includes optimized parameters for the LLM and retriever to ensure faster responses. The frontend uses Streamlit's caching to improve performance.
* **Clear and Informative Responses:** The prompt is engineered to provide answers in a structured, bullet-point format.
* **Graceful Error Handling:** The frontend is designed to manage and display errors that may occur during initialization or response generation.

## Technology Stack

* **Backend:** LangChain, LangGraph, langchain-google-genai
* **Frontend:** Streamlit
* **LLM:** Google Gemini (`gemini-2.0-flash`)
* **Embeddings:** Google Generative AI Embeddings (`models/embedding-001`)
* **Vector Store:** FAISS (`faiss-cpu`)
* **Document Loading:** PyPDF
* **Environment Management:** python-dotenv

## Setup and Installation

Follow these steps to set up and run the project locally.

**1. Clone the Repository**
```bash
git clone <your-repository-url>
cd <your-project-directory>
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

pip install streamlit langchain langchain-openai langgraph faiss-cpu pypdf python-dotenv langchain-google-genai

#run the app
streamlit run frontend.py
