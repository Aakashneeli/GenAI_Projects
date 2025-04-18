from main import LawBot
import streamlit as st
import logging
import webbrowser

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Set the default browser to Chrome
chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
webbrowser.get(chrome_path)

# Initialize the LawBot
try:
    bot = LawBot()
except Exception as e:
    st.error(f"Error initializing LawBot: {e}")
    logging.error(f"Error initializing LawBot: {e}")

# Set up the Streamlit app with improved configuration
st.set_page_config(
    page_title="Indian Constitution Bot",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main app styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: #ffffff;
    }
    
    /* Message styling */
    .message {
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        font-size: 16px;
        line-height: 1.5;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        max-width: 90%;
        font-weight: 500;
    }
    .assistant {
        background-color: #e8f0fe;
        color: #000000;
        border-left: 5px solid #4361ee;
        margin-right: auto;
    }
    .user {
        background-color: #d4edda;
        color: #000000;
        border-right: 5px solid #2ecc71;
        margin-left: auto;
    }
    
    /* Header styling */
    .header {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border-bottom: 3px solid #4361ee;
        color: #000000;
    }
    
    /* Footer styling */
    footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e6e6e6;
        color: #666;
        font-size: 0.8rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Chat input styling */
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced sidebar with more information
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/55/Emblem_of_India.svg", width=100)
    st.title('Indian Constitution Bot')
    st.markdown("### Your AI Legal Assistant")
    st.markdown("---")
    st.markdown("Ask anything regarding the Indian Constitution, laws, rights, and legal procedures in India.")
    
    st.markdown("### Features")
    st.markdown("✅ Constitution knowledge")
    st.markdown("✅ Legal rights information")
    st.markdown("✅ Fundamental duties")
    st.markdown("✅ Government structure")
    
    st.markdown("---")
    st.markdown("*This is an educational tool and not a substitute for professional legal advice.*")

# Header section
st.markdown("<div class='header'><h1 style='text-align: center;'>🇮🇳 Indian Constitution Assistant</h1><p style='text-align: center;'>Ask questions about Indian laws, rights, and constitutional matters</p></div>", unsafe_allow_html=True)

# Function for generating LLM response with improved error handling and caching
@st.cache_data(ttl=3600, show_spinner=False)  # Cache responses for 1 hour
def generate_response(input_text):
    try:
        # Add a progress indicator for better UX during processing
        progress_text = "Searching constitutional knowledge..."
        progress_bar = st.progress(0)
        
        # Simulate progress steps to provide visual feedback
        for i in range(4):
            # Update progress bar
            progress_bar.progress((i + 1) * 25)
            if i == 0:
                st.info("Analyzing your question...")
            elif i == 1:
                st.info("Retrieving relevant constitutional articles...")
            elif i == 2:
                st.info("Formulating response...")
        
        # Get actual response from the bot
        result = bot.rag_chain.invoke(input_text)
        
        # Complete the progress bar and remove it
        progress_bar.progress(100)
        progress_bar.empty()
        
        return result
    except Exception as e:
        st.error(f"Error generating response: {e}")
        logging.error(f"Error generating response: {e}")
        return "Sorry, there was an error processing your request. Please try again later."

# Initialize the session state for storing messages
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "🙏 Namaste! I'm your Indian Constitution Assistant. How can I help you today? You can ask me about fundamental rights, directive principles, constitutional amendments, or any other aspect of the Indian Constitution."
    }]

# Create a container for chat messages
chat_container = st.container()

with chat_container:
    # Display chat messages with improved styling
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            st.markdown(f'<div class="message assistant">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="message user">{message["content"]}</div>', unsafe_allow_html=True)

# User-provided prompt with improved input area
st.markdown("<br>", unsafe_allow_html=True)
user_input = st.chat_input("Type your question about the Indian Constitution here...")

if user_input:
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Update the chat container to show the new message
    with chat_container:
        st.markdown(f'<div class="message user">{user_input}</div>', unsafe_allow_html=True)
    
    # Generate a new response if the last message is from the user
    if st.session_state.messages[-1]["role"] == "user":
        with st.spinner("📚 Consulting the constitution..."):
            response = generate_response(user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Update the chat container to show the new response
            with chat_container:
                st.markdown(f'<div class="message assistant">{response}</div>', unsafe_allow_html=True)

# Adding enhanced footer
st.markdown("<footer><p style='text-align: center;'>🇮🇳 Indian Constitution Bot - Powered by LangChain and Streamlit</p><p style='text-align: center;'>© 2023 - Educational purposes only</p></footer>", unsafe_allow_html=True)
