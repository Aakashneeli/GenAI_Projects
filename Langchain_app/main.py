from dotenv import load_dotenv , find_dotenv
import os

#loading the environment variables
dotenv_path = find_dotenv()

load_dotenv(find_dotenv())

# print(f"Loading .env file from: {dotenv_path}") #debug print

# print(load_dotenv(find_dotenv()))

api_key = os.getenv("OPENAI_API_KEY")

# print(f"API Key from environment: {api_key}")#debug print

if api_key is None:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

from langchain_openai import OpenAI


#Generator 
def generate_pet_name():

    llm = OpenAI(temperature = 0.5,model_name = "gpt-3.5-turbo-instruct", openai_api_key=api_key)
    name = llm.invoke("give me 5 cool pet names")
    return name 

if __name__ == "__main__":
    print(generate_pet_name())



