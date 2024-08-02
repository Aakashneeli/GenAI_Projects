from dotenv import load_dotenv, find_dotenv
import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

class LawBot():
    def __init__(self):
        # Load environment variables
        dotenv_path = find_dotenv()
        load_dotenv(dotenv_path)
        self.api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

        if self.api_key is None:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables")

        # Load PDF
        self.loader = PyPDFLoader("C:\\Users\\Admin\\Documents\\gen_ai_training\\pdfs\\COI.pdf")
        self.docs = self.loader.load()

        # Split documents
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200, add_start_index=True)
        self.chunks = self.text_splitter.split_documents(self.docs)

        # Create embeddings
        modelPath = "sentence-transformers/all-MiniLM-L6-v2"
        self.embeddings = HuggingFaceHubEmbeddings(model=modelPath)

        # Create FAISS index
        self.db = FAISS.from_documents(self.chunks, embedding=self.embeddings)

        # Initialize the LLM
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.llm = HuggingFaceHub(
            repo_id=repo_id, 
            model_kwargs={"temperature": 0.8, "top_k": 50, "min_tokens" : 2048, "max_tokens": 4096}, 
            huggingfacehub_api_token=self.api_key
        )

        # Define the prompt template
        self.template = """
        you are the Indian Constitution and these humans will ask you questions about the law and rights of india
        you will answer them in the form of bullet points which kind of big and informative. 
        the minimum token limit is 2048 so give asnwers according to that.
        Use following piece of context to answer the question.
        If you don't know the answer, give relevant information about the question.

        Context: {context}
        Question: {question}
        Answer: 
        """

        self.prompt = PromptTemplate(
            template=self.template, 
            input_variables=["context", "question"]
        )

        # Define the retriever and RAG chain
        self.retriever = self.db.as_retriever()
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def get_response(self, query):
        result = self.rag_chain.invoke(query)
        # Process result to remove context information
        # Assuming the result is a string where context needs to be stripped off
        answer = result.split("Answer:")[-1].strip()
        return answer

# Instantiate LawBot
bot = LawBot()
user_input = input("Ask me anything about the constitution: ")
result = bot.get_response(user_input)
print(result)





    






