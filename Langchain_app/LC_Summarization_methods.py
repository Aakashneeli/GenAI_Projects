from dotenv import load_dotenv , find_dotenv
import os
dotenv_path = find_dotenv()

load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("OPENAI_API_KEY not found in environment variables")



from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
import glob 
from langchain_openai import OpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz



#StuffDocumentChain
#The chain will take a list of documents, insert them all into a prompt, and pass that prompt to an LLM

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate

#Defining Prompt 
# prompt_template = """Write a concise summary of the following:
# "{text}"
# CONCISE SUMMARY:"""
# prompt = PromptTemplate.from_template(prompt_template)

#defining LLM
# llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o")
# llm_chain = LLMChain(llm=llm, prompt=prompt)

loader  = PyPDFLoader("C:\\Users\\Admin\\Documents\\gen_ai_training\\pdfs\\rag.pdf")
docs = loader.load()

# stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

# print(stuff_chain.invoke(docs)["output_text"])


#Map-Reduce 

# from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
# from langchain_text_splitters import CharacterTextSplitter

# llm = OpenAI(temperature=0.2, model_name="gpt-3.5-turbo")
# # Map Phase
# map_template = """ 
# The following is a set of documents
# {docs}
# Based on this list of docs, please identify the main themes 
# Helpful Answer:
# """
# map_prompt = PromptTemplate.from_template(map_template)
# map_chain = LLMChain(llm=llm, prompt=map_prompt)

# # # Define a reduce function or callable
# # def reduce_function(mapped_outputs):
# #     # Example: Concatenate mapped outputs into a single summary
# #     final_summary = "\n\n".join(mapped_outputs)
# #     return final_summary

# # Reduce Phase
# reduce_template = """  
# The following is a set of summaries:
# {docs}
# Take these and distill it into a final, consolidated summary of the main themes. 
# Helpful Answer:
# """
# reduce_prompt = PromptTemplate.from_template(reduce_template)
# reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# # Combine Map and Reduce Chains using MapReduceDocumentsChain
# map_reduce_chain = MapReduceDocumentsChain(
#     llm_chain=map_chain,  # Use map_chain as the general LLMChain for each document
#     reduce_documents_chain=reduce_chain,
#     document_variable_name="docs",
#     return_intermediate_steps=False  # Adjust as per your requirements
# )
# # Split documents if needed
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# split_docs = text_splitter.split_documents(docs)
# # Run the Map-Reduce Chain
# print(map_reduce_chain.run(split_docs))



from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_text_splitters import CharacterTextSplitter

llm = OpenAI(temperature=0.2, model_name="gpt-3.5-turbo")

# Map
map_template = """The following is a set of documents
{docs}
Based on this list of docs, please identify the main themes 
Helpful Answer:"""
map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm, prompt=map_prompt)
# Reduce
reduce_template = """The following is set of summaries:
{docs}
Take these and distill it into a final, consolidated summary of the main themes. 
Helpful Answer:"""
reduce_prompt = PromptTemplate.from_template(reduce_template)
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)
# Combine documents by mapping a chain over them, then combining results
map_reduce_chain = MapReduceDocumentsChain(
    llm_chain=map_chain,
    reduce_documents_chain=reduce_chain,
    document_variable_name="docs",
    return_intermediate_steps=False,
)
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)
print(map_reduce_chain.run(split_docs))