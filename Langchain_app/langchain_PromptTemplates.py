from dotenv import load_dotenv , find_dotenv
import os
from langchain_openai import OpenAI

#loading the environment variables
dotenv_path = find_dotenv()

load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("OPENAI_API_KEY not found in environment variables")


#promt templates

from langchain import PromptTemplate, LLMChain
llm = OpenAI(temperature = 0.5)

prompt_template  = "Act like a comedian and write a super funny 2 sentence joke about {thing} " 

llm_chain = LLMChain(
    llm = llm ,
    prompt = PromptTemplate.from_template(prompt_template)
)

inputs = [
    {"thing" : "Dads"}, 
    {"thing" : "Asains"},
    {"thing" : "Indians"}
]

response = llm_chain.apply(inputs)

print(response)