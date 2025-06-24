# Hugging Face & LangChain NLP Experiments

This repository contains a collection of Python scripts and Jupyter notebooks created to explore various Natural Language Processing (NLP) tasks using models from the Hugging Face Hub, both directly with the `transformers` library and through the `LangChain` framework.

## Project Overview

This project serves as a personal learning lab for implementing and testing different NLP capabilities. Each script is a self-contained example focusing on a specific task, demonstrating how to load pre-trained models, process data, and generate results.

### Key NLP Tasks Explored:

* **Retrieval-Augmented Generation (RAG)**: An end-to-end RAG pipeline that loads a PDF, creates embeddings, stores them in a FAISS vector store, and uses a `gpt2` model to answer questions based on retrieved context.
* **Text Summarization**: Multiple approaches to summarizing text, including from PDFs using LangChain's `load_summarize_chain` and directly with `transformers` pipelines using models like `google-t5/t5-base` and `MBZUAI/LaMini-Flan-T5-248M`.
* **Question-Answering**: Using pre-trained QA models like `deepset/roberta-base-squad2` to find answers within a given text document.
* **Text Generation**: Generating creative text using models like `distilgpt2`.
* **Translation**: Translating text between languages (e.g., Japanese to English) using models from the Helsinki-NLP group.
* **Sentiment Analysis**: A simple pipeline for classifying the sentiment of a given text.
* **Prompt Templating & Chaining**: Using LangChain to structure complex, multi-step LLM workflows like `RouterChain` and `SequentialChain`.

## Scripts and Notebooks

Here is a breakdown of the key files in this repository:

| File Name                       | Description                                                                                                                                                             |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RAG.ipynb`                     | Implements a full RAG pipeline using Hugging Face embeddings (`sentence-transformers/all-mpnet-base-v2`), a FAISS vector store, and a `gpt2` model for generation.        |
| `Summarization.ipynb`           | A Jupyter Notebook for summarizing text from PDFs using the `MBZUAI/LaMini-Flan-T5-248M` model. It includes preprocessing functions to extract and chunk text.           |
| `TextSummarization.py`          | A script that demonstrates text summarization using the `google-t5/t5-base` model via the `transformers` pipeline.                                                         |
| `QA_model.py`                   | Sets up a Question-Answering pipeline using the `deepset/roberta-base-squad2` model to answer questions from a PDF document.                                              |
| `TextTranslation.py`            | Contains examples of using translation pipelines, including a demonstration of translating Japanese text to English with a `Helsinki-NLP` model.                        |
| `TextGenerator.py`              | A simple script that uses the `distilgpt2` model to generate multiple creative text sequences from a starting prompt.                                                     |
| `SentimentAnalysis.py`          | A basic example of using the `sentiment-analysis` pipeline from the `transformers` library to classify text.                                                              |
| `langchain_RouterChain.py`      | An advanced LangChain script that uses a `MultiPromptChain` to route a user's query to the correct specialized prompt (e.g., for jokes or math problems).                 |
| `langchain_SequentialChains.py` | A script showcasing LangChain's `SimpleSequentialChain` and `SequentialChain` to create multi-step workflows, such as generating a rap and then a diss track.             |
| `langchain_PromptTemplates.py`  | Demonstrates the use of LangChain's `PromptTemplate` to create dynamic prompts, which are then used in an `LLMChain` to generate jokes.                                   |
| `getPdfData.py` & others        | Utility scripts for extracting text from PDF files using libraries like `PyPDF2` and `PyMuPDF (fitz)`.                                                                      |

## Technology Stack

* **Core Libraries**: Hugging Face `transformers`, `LangChain`, `langchain-community`, `langchain-huggingface`
* **Models**: Various models from the Hugging Face Hub (e.g., `gpt2`, `t5-base`, `distilgpt2`, `roberta-base-squad2`, `sentence-transformers`, `Helsinki-NLP`)
* **Vector Store**: FAISS
* **Document Handling**: PyPDFLoader, PyPDF2, PyMuPDF (fitz)
* **Environment Management**: python-dotenv

## Setup and Installation

Follow these steps to set up the environment for running these experiments.

**1. Clone the Repository**

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

pip install transformers torch langchain langchain-community langchain-huggingface python-dotenv pypdf2 pymupdf faiss-cpu sentence-transformers
# .env file

HUGGINGFACEHUB_API_TOKEN="your-hugging-face-api-token"
OPENAI_API_KEY="your-openai-api-key"
