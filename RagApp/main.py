# from dotenv import load_dotenv , find_dotenv
# import os
# dotenv_path = find_dotenv()

# load_dotenv(find_dotenv())
# api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# if api_key is None:
#     raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables")


from typing import List, Dict
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers import MarianMTModel, MarianTokenizer
import os

# Constants
MODEL_NAME = "google/flan-t5-base"
TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-en-es"
MAX_LENGTH = 300
MIN_LENGTH = 100
TEMPERATURE = 0.7
TOP_P = 0.95

DETAILED_PROMPT = """
You are an AI assistant tasked with providing detailed and comprehensive answers based on the given context. Your responses should be thorough, well-structured, and informative.

Context: {context}
"""

def initialize_models():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        min_length=MIN_LENGTH,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        num_return_sequences=1,
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    embeddings = HuggingFaceEmbeddings()
    
    translation_tokenizer = MarianTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
    translation_model = MarianMTModel.from_pretrained(TRANSLATION_MODEL_NAME)
    
    return llm, embeddings, translation_tokenizer, translation_model

def process_pdf(pdf_path: str, embeddings) -> FAISS:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return FAISS.from_documents(texts, embeddings)

def query_document(db: FAISS, query: str, llm, prompt_template: str = DETAILED_PROMPT) -> str:
    prompt = PromptTemplate(input_variables=["context"], template=prompt_template)
    formatted_prompt = prompt.format(context=query)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
    result = qa_chain({"query": formatted_prompt})
    return result['result']

def summarize_document(db: FAISS, llm) -> str:
    summary_prompt = "Provide a concise summary of the main points in the document."
    return query_document(db, summary_prompt, llm)

def translate_to_spanish(text: str, translation_tokenizer, translation_model) -> str:
    inputs = translation_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated = translation_model.generate(**inputs)
    return translation_tokenizer.decode(translated[0], skip_special_tokens=True)

def get_db_from_file(file_path: str, embeddings) -> FAISS:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return process_pdf(file_path, embeddings)

# Initialize models and embeddings
llm, embeddings, translation_tokenizer, translation_model = initialize_models()

