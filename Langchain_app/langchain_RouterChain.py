from dotenv import load_dotenv , find_dotenv
import os
from langchain_openai import OpenAI

#loading the environment variables
dotenv_path = find_dotenv()

load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Router chains

from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from langchain.chains.router import MultiPromptChain
from langchain.chains import ConversationChain

llm = OpenAI(temperature = 0.5)

#Joke Template 
joke_template = "Act like a comedian and write a super funny 2 sentence joke about {input} "
joke_prompt = PromptTemplate.from_template(joke_template)
joke_chain = LLMChain(llm = llm , prompt = joke_prompt)

#Math Template 
math_template = "Act like an experienced math Professor who gets right to the point about solving {input}"
math_prompt = PromptTemplate.from_template(math_template)
math_chain = LLMChain(llm = llm , prompt = math_prompt)

prompt_infos = [
    {
        "name": "joke",
        "description": "Good for generating jokes",
        "prompt_template": joke_template,
    },
    {
        "name": "math",
        "description": "Good for solving math problems",
        "prompt_template": math_template,
    },
]

destination_chain = {
    "joke": joke_chain,
    "math": math_chain,
}

#define a default chain for general questions
default_chain = ConversationChain(llm = llm) 


# Create the MultiPromptChain
multi_prompt_chain = MultiPromptChain.from_prompts(
    prompt_infos=prompt_infos,
    llm=llm,
    default_chain=default_chain
)


#defining inputs :
inputs = [
    {"input" : "Tell me a joke about Americans"}, 
    {"input" : "Tell me the answer to this math problem : what is the square root of 1976"},
    {"input" : "my friend wants to hear a nice joke , tell me very funny one."}
]

for input_data in inputs:
    response = multi_prompt_chain(input_data)
    print(response, end="\n")