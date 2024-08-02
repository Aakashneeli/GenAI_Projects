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

# Set up the Streamlit app
st.set_page_config(page_title="Indian Constitution Bot", layout="wide")
with st.sidebar:
    st.title('Indian Constitution Bot')
    st.write("Ask anything regarding the Indian Constitution.")

# Function for generating LLM response
def generate_response(input_text):
    try:
        result = bot.rag_chain.invoke(input_text)
        return result
    except Exception as e:
        st.error(f"Error generating response: {e}")
        logging.error(f"Error generating response: {e}")
        return "Sorry, there was an error processing your request."

# Initialize the session state for storing messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome, ask anything regarding our constitution."}]

# Display chat messages
st.markdown("""
    <style>
        .message {border-radius: 10px; padding: 10px; margin: 5px 0; font-size: 16px;}
        .assistant {background-color: #f0f0f5; color: #333333;}
        .user {background-color: #dff9fb; color: #333333;}
    </style>
""", unsafe_allow_html=True)

for message in st.session_state.messages:
    if message["role"] == "assistant":
        st.markdown(f'<div class="message assistant">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="message user">{message["content"]}</div>', unsafe_allow_html=True)

# User-provided prompt
user_input = st.chat_input("Type your question here...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f'<div class="message user">{user_input}</div>', unsafe_allow_html=True)

    # Generate a new response if the last message is from the user
    if st.session_state.messages[-1]["role"] == "user":
        with st.spinner("Getting your answer from the constitution..."):
            response = generate_response(user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(f'<div class="message assistant">{response}</div>', unsafe_allow_html=True)

# Adding footer
st.markdown("<footer><p style='text-align: center;'>Indian Constitution Bot - Powered by LangChain and Streamlit</p></footer>", unsafe_allow_html=True)
