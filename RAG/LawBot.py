import os
import streamlit as st
from dotenv import load_dotenv

# LangChain and LangGraph imports
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

# --- App Configuration ---
st.set_page_config(page_title="Lawbot", page_icon="⚖️", layout="wide")

# Load environment variables from a .env file
load_dotenv()

st.title("⚖️ Lawbot: Your AI Legal Assistant")
st.write("Answering your questions about Indian law and the Constitution.")
st.info("This app is configured to use the `COI.pdf` file and your `OPENAI_API_KEY` from the environment.")

# --- Core Application Logic ---

# Function to initialize models, cached for efficiency
@st.cache_resource
def get_models():
    """Initializes and returns the LLM and embeddings model."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not found in environment variables. Please create a .env file.")
        return None, None
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
        return llm, embeddings
    except Exception as e:
        st.error(f"Error initializing OpenAI models: {e}")
        st.error("Please check your API key and model access permissions.")
        return None, None

# Function to create the vector store from the local PDF file
@st.cache_resource
def create_vector_store(_embeddings):
    """Creates a FAISS vector store from the local PDF file."""
    pdf_path = "COI.pdf"  # Using a local PDF file
    if not _embeddings:
        return None
    if not os.path.exists(pdf_path):
        st.error(f"The document '{pdf_path}' was not found. Please place it in the same directory as the app.")
        return None
    try:
        # Load the document
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        if not docs:
            st.warning("Could not load any documents from the PDF.")
            return None

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)
        if not all_splits:
            st.warning("Could not split the document into chunks.")
            return None
        
        # Create and return the FAISS vector store
        st.success(f"Document '{pdf_path}' processed successfully. Ready for queries.")
        return FAISS.from_documents(documents=all_splits, embedding=_embeddings)

    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

# Main app flow
llm, embeddings = get_models()
if llm and embeddings:
    vector_store = create_vector_store(embeddings)

    if vector_store:
        
        # --- Tool Definition ---
        @tool
        def retrieve(query: str) -> tuple[str, list]:
            """Retrieve relevant document chunks based on the user query from the Indian legal document."""
            retrieved_docs = vector_store.similarity_search(query, k=5)
            if not retrieved_docs:
                return "No relevant documents found.", []
            context = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)
            return context, retrieved_docs

        # --- Graph Definition ---
        @st.cache_resource
        def create_graph(_llm):
            """Creates and compiles the LangGraph agent."""
            # Define nodes
            def query_or_respond(state: MessagesState):
                llm_with_tools = _llm.bind_tools([retrieve])
                response = llm_with_tools.invoke(state['messages'])
                return {"messages": [response]}

            tools_node = ToolNode([retrieve])

            def generate(state: MessagesState):
                messages = state['messages']
                
                # Correct way to get context from the last ToolMessage
                retrieved_context = ""
                for msg in reversed(messages):
                    if isinstance(msg, ToolMessage):
                        # The tool's output tuple is stored in the .content attribute
                        if isinstance(msg.content, str): # Handle if tool returns string error
                            retrieved_context = msg.content
                        break

                system_prompt = (
                    "You are Lawbot, an expert assistant for answering Indian Legal & Constitutional queries. "
                    "Use the following pieces of retrieved context ONLY if they are relevant to the user's latest question. "
                    "If the context is provided but irrelevant, ignore it. "
                    "If you don't know the answer based on the conversation history and relevant context, state that clearly. "
                    "Answer the user's LATEST question based on the full conversation history and any relevant retrieved context. "
                    "Keep the answer concise and clear."
                )

                if retrieved_context and "No relevant documents found" not in retrieved_context:
                    system_prompt += (
                        "\n\n--- Retrieved Context ---\n"
                        f"{retrieved_context}"
                        "\n--- End Context ---"
                    )

                prompt_messages = [SystemMessage(content=system_prompt)]
                # Add all messages except the tool-related ones for the final generation
                for msg in messages:
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        continue # Don't include the AI message that called the tool
                    if isinstance(msg, ToolMessage):
                        continue # Don't include the raw tool output
                    prompt_messages.append(msg)
                
                final_response = _llm.invoke(prompt_messages)
                return {"messages": [final_response]}
            
            # Build graph
            graph_builder = StateGraph(MessagesState)
            graph_builder.add_node("query_or_respond", query_or_respond)
            graph_builder.add_node("retrieve_tool", tools_node)
            graph_builder.add_node("generate", generate)
            graph_builder.set_entry_point("query_or_respond")
            graph_builder.add_conditional_edges(
                "query_or_respond",
                tools_condition,
                {"tools": "retrieve_tool", END: "generate"},
            )
            graph_builder.add_edge("retrieve_tool", "generate")
            graph_builder.add_edge("generate", END)
            return graph_builder.compile()

        graph = create_graph(llm)

        # --- Chat Interface ---
        # Initialize session state for messages if it doesn't exist
        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.langgraph_history = []

        # Display past messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Get user input
        if prompt := st.chat_input("Ask a question about the document..."):
            # Add user message to session state and display it
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Prepare graph input
            graph_input = {"messages": st.session_state.langgraph_history + [HumanMessage(content=prompt)]}
            
            # Stream the response from the graph
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                final_state = None
                # Use stream_mode="values" to get the state at each step
                for step in graph.stream(graph_input, {"recursion_limit": 10}, stream_mode="values"):
                    final_state = step

                # The final AIMessage in the history is the answer
                if final_state and final_state.get("messages"):
                    ai_message = final_state["messages"][-1]
                    if isinstance(ai_message, AIMessage):
                       full_response = ai_message.content

                message_placeholder.markdown(full_response)
            
            # Add the final AI response and the full LangGraph history to session state
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.langgraph_history = final_state["messages"] if final_state else []
