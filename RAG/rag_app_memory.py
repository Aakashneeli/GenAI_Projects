import os
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage, AIMessage # Specific message types
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI # Direct OpenAI classes
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss # Vector database library
from langgraph.graph import MessagesState, StateGraph, END # LangGraph core components
from langgraph.prebuilt import ToolNode, tools_condition # LangGraph helpers
from IPython.display import Image, display # For visualization
import getpass # Securely get password (optional fallback for API key)
from typing import Dict, List, TypedDict, Sequence # For typing MessagesState explicitly if needed

# Define the structure of the state for clarity (optional but good practice)
# class AgentState(TypedDict):
#     messages: Sequence[BaseMessage]
# Using MessagesState directly is often sufficient as it implies {"messages": Sequence[BaseMessage]}


# --- Configuration & Setup ---
# 1. Load Environment Variables (API Keys)
dotenv_path = find_dotenv() # Searches for a .env file in parent directories
if dotenv_path:
    print(f"Loading .env file from: {dotenv_path}")
    load_dotenv(dotenv_path) # Loads variables from .env into environment variables
else:
    print(".env file not found. Ensure API keys are set as environment variables.")

# 2. Retrieve API Keys
api_key = os.getenv("OPENAI_API_KEY") # Get OpenAI key from environment
langsmith_key = os.getenv("LANGSMITH_API_KEY") # Get LangSmith key (optional, for tracing/debugging)

# 3. Configure LangSmith (Optional Tracing)
if langsmith_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true" # Enable LangSmith tracing
    os.environ["LANGCHAIN_API_KEY"] = langsmith_key # Provide the LangSmith key
    # os.environ["LANGCHAIN_PROJECT"] = "Your Project Name" # Optional: Organize runs

# 4. Validate OpenAI API Key
if not api_key:
    # You could add a fallback like getpass.getpass here, but raising an error is often safer
    raise ValueError("OPENAI_API_KEY not found in environment variables or .env file.")

# --- LLM and Embeddings ---
# 5. Initialize the Language Model (LLM)
try:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key
    )
# 6. Initialize the Embedding Model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=api_key
    )
except Exception as e:
    print(f"Error initializing OpenAI models: {e}")
    print("Please check your API key and model access permissions.")
    exit()

# --- Vector Store Setup ---
# 7. Prepare for FAISS (Vector Database)
try:
    test_embedding = embeddings.embed_query("hello world")
    embedding_dim = len(test_embedding)
    print(f"Embedding dimension: {embedding_dim}")
except Exception as e:
    print(f"Error getting embedding dimension: {e}")
    exit()

# 8. Create a FAISS Index
index = faiss.IndexFlatL2(embedding_dim)

# 9. Create an In-Memory Document Store
docstore = InMemoryDocstore()

# 10. Create the FAISS Vector Store Wrapper
vector_store = FAISS(
    embedding_function=embeddings.embed_query, # Use embed_query for single queries
    index=index,
    docstore=docstore,
    index_to_docstore_id={},
)

# --- Document Loading and Processing ---
# 11. Define PDF Path and Load Document
pdf_path = "C:\\Users\\Admin\\Documents\\gen_ai_training\\pdfs\\rag.pdf" # <<< MAKE SURE THIS PATH IS CORRECT FOR YOUR SYSTEM
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

try:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from PDF.")

    # 12. Split Documents into Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"Split document into {len(all_splits)} chunks.")

    # 13. Add Chunks to Vector Store (Embedding and Indexing)
    if all_splits:
        print("Adding documents to vector store...")
        # Use add_documents which internally handles embedding list of docs
        vector_store.add_documents(documents=all_splits)
        print("Documents added successfully.")
    else:
        print("Warning: No text chunks were generated from the PDF.")

except Exception as e:
    print(f"Error loading or processing PDF: {e}")
    exit()

# --- Tool Definition ---
# 14. Define the Retrieval Tool
@tool
def retrieve(query: str) -> tuple[str, list]:
    """Retrieve relevant document chunks based on the user query."""
    print(f"--- Executing Retrieval for query: '{query}' ---")
    try:
        # 15. Perform Similarity Search
        retrieved_docs = vector_store.similarity_search(query, k=3)

        if not retrieved_docs:
            print("--- No documents found by retriever. ---")
            return "No relevant documents found.", []

        # 16. Format Retrieved Context for LLM
        context = "\n\n---\n\n".join(
            f"Content: {doc.page_content}"
            for doc in retrieved_docs
        )
        print(f"--- Retrieved {len(retrieved_docs)} documents. ---")

        # 17. Return Formatted Context and Raw Documents (as tuple)
        return context, retrieved_docs
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return f"Error during retrieval: {e}", []

# --- Graph Node Definitions ---

# 18. Define the Router Node (`query_or_respond`)
def query_or_respond(state: MessagesState):
    """Decide whether to call the retrieval tool or respond directly."""
    print("--- Node: query_or_respond ---")
    messages = state['messages']
    print(f"Router received {len(messages)} messages.")

    # 19. Give LLM the Option to Use the Tool
    llm_with_tools = llm.bind_tools([retrieve])

    # 20. Invoke LLM for Decision
    # Pass the entire message history for context
    response = llm_with_tools.invoke(messages)
    print(f"Response from query_or_respond LLM: {response.pretty_repr()}")

    # 21. Update State
    # LangGraph MessagesState automatically appends this response to the list
    return {"messages": [response]}

# 22. Define the Tool Execution Node
tools_node = ToolNode([retrieve]) # Handles execution based on AIMessage tool_calls

# 23. Define the Generation Node (`generate`)
def generate(state: MessagesState):
    """Generate a final response using the conversation history and retrieved context."""
    print("--- Node: generate ---")
    messages = state['messages']
    print(f"Generator received {len(messages)} messages.")

    # 24. Extract Tool Results (Context)
    tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]

    # 25. Process Tool Results (Context Extraction)
    retrieved_context = "\n\n".join(
        msg.content[0] # Access the FIRST element (context string) of the tuple
        for msg in tool_messages
        if isinstance(msg.content, tuple) and len(msg.content) > 0 and isinstance(msg.content[0], str)
    )

    # 26. Prepare Prompt for Final Answer Generation
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context ONLY if they are relevant to the user's latest question. "
        "If the context is provided but irrelevant, ignore it. "
        "If you don't know the answer based on the conversation history and relevant context, say that you don't know. "
        "Answer the user's LATEST question based on the full conversation history and any relevant retrieved context. "
        "Keep the answer concise."
    )

    if retrieved_context:
        print(f"--- Context for generation:\n{retrieved_context[:500]}... ---")
        system_prompt += (
            "\n\n--- Retrieved Context ---\n"
            f"{retrieved_context}"
            "\n--- End Context ---"
        )
    else:
        print("--- No context found from retrieval tool for this turn. Generating response based on history only. ---")

    # 27. Prepare message list for final LLM call
    # Filter out ToolMessages and the AI message that initiated the tool call,
    # then add the system prompt.
    prompt_messages = [SystemMessage(content=system_prompt)]
    for msg in messages:
        if isinstance(msg, ToolMessage):
            continue # Skip tool results
        if isinstance(msg, AIMessage) and msg.tool_calls:
            continue # Skip the AI message that made the tool call
        prompt_messages.append(msg) # Keep user messages and previous final AI answers

    # 29. Invoke LLM for Final Answer
    final_response = llm.invoke(prompt_messages)
    print(f"Final generated response: {final_response.pretty_repr()}")

    # 30. Update State with Final Answer
    return {"messages": [final_response]}

# --- Graph Construction ---
# 31. Initialize StateGraph with MessagesState
graph_builder = StateGraph(MessagesState)

# 32. Add Nodes
graph_builder.add_node("query_or_respond", query_or_respond)
graph_builder.add_node("retrieve_tool", tools_node)
graph_builder.add_node("generate", generate)

# 33. Set Entry Point
graph_builder.set_entry_point("query_or_respond")

# 34. Add Conditional Edges
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition, # Checks if the last message contains tool_calls
    {
        "tools": "retrieve_tool", # If tool_calls detected, go to retrieve_tool
        END: END                  # Otherwise (no tool_calls), end the graph run
    }
)

# 35. Add Normal Edges
graph_builder.add_edge("retrieve_tool", "generate") # After retrieving, always generate
graph_builder.add_edge("generate", END)             # After generating, always end

# 36. Compile the Graph
graph = graph_builder.compile()

# --- Visualization (Optional) ---
# 37. Draw the Graph Structure
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"Could not display graph visualization: {e}. Ensure graphviz and pygraphviz are installed if needed.")

# --- Running the Graph with Persistent Memory ---

# 38. Define Helper Function (MODIFIED FOR MEMORY)
def run_query_with_memory(input_message: str, current_state: MessagesState) -> MessagesState:
    """
    Runs the graph with the given input message, using and updating the provided conversation state.

    Args:
        input_message: The new message from the user.
        current_state: The current state dictionary (containing the 'messages' list).

    Returns:
        The final state dictionary after the graph execution finishes, containing the full updated history.
    """
    print(f"\n--- Running Query: '{input_message}' ---")
    print(f"--- Input State Messages Count: {len(current_state.get('messages', []))} ---")

    # 39. Prepare Graph Input with Full History
    # Append the new user message to the existing list of messages
    messages_history = current_state.get("messages", [])
    graph_input = {"messages": messages_history + [HumanMessage(content=input_message)]}

    final_state = None # Initialize variable to hold the very last state

    # 40. Stream Graph Execution (using graph_input with history)
    for step in graph.stream(
        graph_input,
        {"recursion_limit": 10}, # Add recursion limit for safety
        stream_mode="values",
    ):
        print("\n--- State Update ---")
        # Print the last message added in this step
        last_message = step["messages"][-1]
        last_message.pretty_print()
        final_state = step # Keep track of the latest complete state dictionary

    # 41. Print Final Answer Information (from the last step)
    print("\n--- Final Message in This Run ---")
    if final_state and final_state.get("messages"):
        final_message = final_state["messages"][-1]
        # Check if it's a standard AI response (likely the end of this turn)
        if isinstance(final_message, AIMessage) and not final_message.tool_calls:
            print("(This looks like the final answer for this turn)")
            # final_message.pretty_print() # Already printed in the loop
        else:
            print(f"(Last message type in run: {type(final_message).__name__})")
    else:
        print("No final state or messages found in the last step.")

    print("--- Query Finished ---")

    # 42. Return the FINAL state dictionary containing the full history
    return final_state if final_state is not None else current_state


# --- Main Execution Logic with Memory ---

# 43. Initialize Conversation State (Starts Empty)
# This dictionary will be updated after each interaction.
conversation_state: MessagesState = {"messages": []}

# 44. Run Test Cases (Updating conversation_state each time)

# Interaction 1
print("\n\n====================== Interaction 1 ======================")
conversation_state = run_query_with_memory("Hello, how are you today?", conversation_state)
# 'conversation_state' now holds: [HumanMessage("Hello..."), AIMessage("I'm just a bot...")]

# Interaction 2
print("\n\n====================== Interaction 2 ======================")
conversation_state = run_query_with_memory("Based on the document you have, what is Naive RAG?", conversation_state)
# 'conversation_state' now holds messages from Interaction 1 +
# [HumanMessage("...native RAG?"), AIMessage(tool_calls=[...]), ToolMessage(...), AIMessage("Native RAG is...")]
# The LLM nodes had access to the first interaction when processing the second.

# Interaction 3
print("\n\n====================== Interaction 3 ======================")
conversation_state = run_query_with_memory("And what are its main benefits?", conversation_state)
# 'conversation_state' holds all previous messages. The LLM uses this full history
# to understand "its" refers to "native RAG" and answer the question, possibly using retrieval again.

# Interaction 4 (Example of potentially no retrieval needed)
print("\n\n====================== Interaction 4 ======================")
conversation_state = run_query_with_memory("Thanks!", conversation_state)
# The LLM likely won't call the tool here, query_or_respond goes directly to END.

# 45. Inspect Final Conversation State (Optional)
print("\n\n====================== Final Conversation State ======================")
print(f"Total messages in history: {len(conversation_state.get('messages', []))}")
