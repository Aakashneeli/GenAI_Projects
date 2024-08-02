from dotenv import load_dotenv , find_dotenv
import os
from langchain_openai import OpenAI

#loading the environment variables
dotenv_path = find_dotenv()

load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("OPENAI_API_KEY not found in environment variables")


#LLM wrappers
from langchain.schema import(
    AIMessage, 
    HumanMessage, 
    SystemMessage
)

from langchain_community.chat_models import ChatOpenAI


chat = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0.4)
messages = [
    SystemMessage(content="You are an expert data scientist"),
    HumanMessage(content="write a python script for implementing a pipeline which cleans and transforms the data?"),
  
]
response = chat(messages)

print(response.content, end = "\n")
