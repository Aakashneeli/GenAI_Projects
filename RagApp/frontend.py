import streamlit as st
import os
import tempfile
from main import get_db_from_file, query_document, summarize_document, translate_to_spanish, initialize_models

st.set_page_config(page_title="PDF Q&A Chatbot", layout="wide")

@st.cache_resource
def load_models():
    return initialize_models()

@st.cache_resource
def load_db(file_path, _embeddings):
    return get_db_from_file(file_path, _embeddings)
def main():
    st.title("PDF Q&A Chatbot")

    llm, embeddings, translation_tokenizer, translation_model = load_models()

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        with st.spinner("Processing PDF..."):
            db = load_db(tmp_file_path, embeddings)
            st.success("PDF processed successfully!")

        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Document Summary")
            if st.button("Summarize"):
                with st.spinner("Generating summary..."):
                    summary = summarize_document(db, llm)
                    st.write(summary)
        
        with col2:
            st.subheader("Ask a Question")
            question = st.text_input("Enter your question:")
            if st.button("Submit Question"):
                if question:
                    with st.spinner("Finding answer..."):
                        answer = query_document(db, question, llm)
                        st.write("Answer:", answer)
                else:
                    st.warning("Please enter a question.")
        
        with col3:
            st.subheader("Translate to Spanish")
            text_to_translate = st.text_area("Enter text to translate:")
            if st.button("Translate"):
                if text_to_translate:
                    with st.spinner("Translating..."):
                        translated = translate_to_spanish(text_to_translate, translation_tokenizer, translation_model)
                        st.write("Translated text:")
                        st.write(translated)
                else:
                    st.warning("Please enter text to translate.")

        # Clean up the temporary file
        os.unlink(tmp_file_path)

    else:
        st.info("Please upload a PDF file to get started.")

    st.sidebar.title("About")
    st.sidebar.info("This chatbot uses RAG with a GPT model to answer questions about uploaded PDF documents.")

if __name__ == "__main__":
    main()