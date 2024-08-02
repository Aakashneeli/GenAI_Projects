from dotenv import load_dotenv , find_dotenv
import os
from langchain_openai import OpenAI

#loading the environment variables
dotenv_path = find_dotenv()

load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Sequential chains takes the output from one call and use it as the input to another. 
# Sequential chains allow you to connect multiple chains and compose them into pipelines executing a specific scenario.
# 2 types : SimpleSeqentialChain, SequentialChain

from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain


llm = OpenAI(temperature = 0.5)

#SimpleSequentialChain
#This is a prompt to write a rap    
rap_template = """

You are a  rapper, like Eminem or J Cole.

Given a topic, it is your job to spit bars on of pure heat.

Topic: {topic}
"""

prompt_template = PromptTemplate(input_variables=["topic"], template=rap_template)

rap_chain = LLMChain(llm=llm, prompt=prompt_template)

# This is an LLMChain to write a diss track

llm = OpenAI(temperature=.7)

diss_template = """

You are an extremely competitive Rapper.

Given the rap from another rapper, it's your job to write a diss track which
tears apart the rap and shames the original rapper.

Rap:
{rap}
"""

prompt_template = PromptTemplate(input_variables=["rap"], template=diss_template)

diss_chain = LLMChain(llm=llm, prompt=prompt_template)

#This is the overall chain where we run these two chains in sequence

overall_chain1 = SimpleSequentialChain(chains=[rap_chain, diss_chain], verbose=True)

disstrack = overall_chain1.run("Drinking Crown Royal and mobbin in my red Challenger")

# print(disstrack)

#SequentialChain

#Verses Prompt
verses_template = """
you are a skilled singer and song writer and it is your job to create a rhyme of two verses and one chorus
for each topic.

Topic = {topic1} and {topic2}

song:

"""

verses_prompt_template = PromptTemplate(input_variables = ["topic1", "topic2"], template = verses_template)

verses_chain = LLMChain(llm = llm , prompt = verses_prompt_template, output_key = "song")

#review template

review_template = """
You are a rap critic from the Rolling Stone magazine and Metacritic.

Given a, it is your job to write a review for that song.

Your review style should be scathing, critical, and no holds barred.

Song:

{song}

Review from the Rolling Stone magazine and Metacritic critic of the above rap:

"""

review_prompt_template = PromptTemplate(input_variables = ["song"], template = review_template)

review_chain = LLMChain(llm = llm , prompt = review_prompt_template, output_key  = "review")

#overall chain where we'll run these 2 chains in sequence 


overall_chain2 = SequentialChain(
    chains=[verses_chain , review_chain],
    input_variables=["topic1", "topic2"],
    # Here we return multiple variables
    output_variables=["song", "review"],
    verbose=True)

Response = overall_chain2({"topic1" : "desire", "topic2" : "Mountains"})


print("Generated Song:\n")
print(Response['song'])
print("\nReview:\n")
print(Response['review'])

