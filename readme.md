# AI Agent Builder

This project is a conversational AI agent that builds other AI agents. A user can describe the agent they want to create, and this system will reason about the requirements, consult relevant documentation, and generate the complete, production-ready code files for the new agent.

The entire application is orchestrated by LangGraph, with a user interface provided by Streamlit.

## How It Works

The AI Agent Builder uses a multi-agent system to handle a user's request. The workflow is as follows:

1.  **User Request**: The user enters a prompt in the Streamlit UI describing the AI agent they want to build (e.g., "Build me an AI agent that can search the web with the Brave API.").
2.  **Define Scope**: A specialized **Reasoner Agent** (`o3-mini`) analyzes the user's request. It consults a list of available documentation pages from a Supabase vector store to understand the context and creates a detailed scope document, including architecture, components, and a testing strategy. This scope is saved to a `workbench/scope.md` file.
3.  **Code Generation**: The primary **Coder Agent** (`gpt-4o`), which is an expert in `Pydantic AI`, takes the scope document and the user's conversation history. It uses its built-in tools to perform Retrieval-Augmented Generation (RAG) against the documentation in the Supabase database to find relevant examples and API references.
4.  **File Creation**: Based on its research, the Coder Agent writes the code for the new agent into a set of standard files (`agent.py`, `agent_tools.py`, `agent_prompts.py`, `.env.example`, `requirements.txt`).
5.  **Conversational Flow**: After each turn, the graph is interrupted to get feedback or further instructions from the user. A **Router Agent** then intelligently decides whether to continue with more coding or to end the conversation based on the user's input.
6.  **Completion**: When the user is satisfied, a final agent provides instructions on how to run the newly created agent code.

The entire process is managed as a state machine by a `LangGraph` graph, which handles transitions between the different agents and tools.

## Key Components

-   **`streamlit_ui.py`**: The main entry point for the user. It builds the web-based chat interface using Streamlit and handles the streaming of responses from the agent graph.
-   **`agent_graph.py`**: Defines the core agentic workflow using `StateGraph`. It sets up the nodes (agents, tools) and edges (transitions) that orchestrate the entire agent-building process. It also configures memory persistence to maintain conversation state.
-   **`pydantic_ai_agent_coder.py`**: Contains the implementation of the main `pydantic_ai_coder` agent and its associated tools. These tools allow the agent to perform RAG by searching and retrieving content from the documentation stored in the vector database.
-   **`crawl_pydantic_ai_docs.py`**: A data ingestion script responsible for populating the knowledge base. It crawls the Pydantic AI documentation, chunks the content, generates embeddings, and stores it all in the Supabase database.
-   **`site_pages.sql`**: The SQL schema for the PostgreSQL database. It defines the `site_pages` table, which includes a `vector` column for embeddings, and creates the `match_site_pages` function for performing vector similarity search.

## Technology Stack

-   **Frontend**: Streamlit
-   **Backend Orchestration**: LangGraph
-   **Core Agent Logic**: Pydantic AI
-   **LLMs**: OpenAI (`gpt-4o`), o3-mini
-   **Embeddings**: OpenAI (`text-embedding-3-small`)
-   **Vector Database**: PostgreSQL with `pgvector` extension (hosted on Supabase)
-   **Web Scraping**: Crawl4AI
-   **Environment Management**: python-dotenv

## Setup and Installation

Follow these steps to set up the project locally.

### Prerequisites

-   A PostgreSQL database with the `pgvector` extension enabled. [Supabase](https://supabase.com/) is a recommended and easy-to-use option.
-   Python 3.9+

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-project-directory>


# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

pip install -r requirements3.txt


# .env file

# Supabase Credentials
SUPABASE_URL="your-supabase-project-url"
SUPABASE_SERVICE_KEY="your-supabase-service-role-key"

# OpenAI API Key (used by multiple components)
OPENAI_API_KEY="your-openai-api-key"

# Optional: For configuring different models
PRIMARY_MODEL="gpt-4o"
REASONER_MODEL="o3-mini"
EMBEDDING_MODEL="text-embedding-3-small"


#Run the crawler
python crawl_pydantic_ai_docs.py

#run the streamlit app
streamlit run streamlit_ui.py
