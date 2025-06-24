# LangChain Experiments & Learning Lab

This repository is a personal collection of Python scripts and Jupyter notebooks created to explore and understand the core components of the [LangChain](https://www.langchain.com/) framework. Each file focuses on a specific feature, demonstrating its implementation and usage with the OpenAI API.

## Project Overview

The primary goal of this repository is to serve as a hands-on learning environment for various LangChain concepts. The scripts cover fundamental building blocks such as LLM wrappers, prompt templating, document handling, and different types of chains.

### Key Concepts Explored:

* **LLM Wrappers**: Interfacing with chat models like `gpt-3.5-turbo`.
* **Prompt Templating**: Creating dynamic and reusable prompts for different tasks.
* **Document Loading**: Extracting text content from PDF files using multiple methods.
* **Document Summarization**: Implementing the `MapReduce` technique to summarize large documents.
* **Chaining**: Building complex workflows by connecting multiple LLM calls.
    * **SimpleSequentialChain**: A basic chain for linear, single-input/output sequences.
    * **SequentialChain**: A more advanced chain that handles multiple inputs and outputs between calls.
    * **RouterChain**: An intelligent chain that dynamically selects the next chain to execute based on the input.

## Scripts and Notebooks

Here is a breakdown of each file in this repository:

| File Name                       | Description                                                                                                                                            |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `langchain_llm_wrapper.py`      | Demonstrates how to wrap and interact with an OpenAI chat model (`gpt-3.5-turbo`) using LangChain's `SystemMessage` and `HumanMessage` schema.              |
| `langchain_PromptTemplates.py`  | Shows how to create and use `PromptTemplate` and `LLMChain` to generate jokes on various topics by acting as a comedian.                               |
| `langchain_SequentialChains.py` | Implements both `SimpleSequentialChain` and `SequentialChain`. The first creates a rap and a diss track, while the second creates a song and a critic's review. |
| `langchain_RouterChain.py`      | Implements a `MultiPromptChain` that acts as a router. It intelligently directs a user's query to either a "joke" chain or a "math" chain based on the input. |
| `langchain_Summarization.ipynb` | A Jupyter notebook that summarizes all PDFs within a specified folder using the `load_summarize_chain` with a `map_reduce` strategy. It also shows how to query the indexed documents. |
| `getPdfData.py`                 | A utility script providing multiple functions (`langchainPyPDFLoader`, `pyPdfReaderFun`) to demonstrate different methods of extracting raw text from PDF files. |

*(Note: Some scripts may have duplicate names or content, reflecting different stages of experimentation.)*

## Technology Stack

* **Core Framework**: LangChain, langchain-openai
* **LLMs**: OpenAI (GPT-3.5 Turbo, etc.)
* **Document Handling**: PyPDF2, PyMuPDF (fitz), PyPDFDirectoryLoader
* **Environment Management**: python-dotenv

## Setup and Installation

Follow these steps to set up the environment and run the scripts.

**1. Setting up**

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# .env file
OPENAI_API_KEY="your-openai-api-key-here"

#running a script 
python langchain_SequentialChains.py
