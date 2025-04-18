from dotenv import load_dotenv, find_dotenv
import os
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class LawBot():
    def __init__(self):
        # Load environment variables
        dotenv_path = find_dotenv()
        load_dotenv(dotenv_path)
        self.google_api_key = os.getenv("GOOGLE_API_KEY")

        if self.google_api_key is None:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
        # Configure Google Generative AI with API key
        genai.configure(api_key=self.google_api_key)

        # Load PDF
        self.loader = PyPDFLoader("C:\\Users\\Admin\\Documents\\gen_ai_training\\pdfs\\COI.pdf")
        self.docs = self.loader.load()

        # Split documents - smaller chunks for faster processing
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
        self.chunks = self.text_splitter.split_documents(self.docs)

        # Create embeddings using Google's embedding model instead of HuggingFace
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.google_api_key,
            task_type="retrieval_query"
        )

        # Create FAISS index with optimized parameters
        self.db = FAISS.from_documents(self.chunks, embedding=self.embeddings)

        # Initialize the Gemini Pro LLM with optimized parameters for faster responses
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.4,  # Lower temperature for more focused responses
            top_p=0.85,
            top_k=30,
            max_output_tokens=800,  # Reduced token count for faster generation
            google_api_key=self.google_api_key
        )

        # Define the prompt template
        self.template = """
        You are an expert on the Indian Constitution. Your task is to answer questions about Indian law and rights.
        Format your answers as informative bullet points.
        Provide comprehensive information based on the context provided.
        If you don't know the exact answer, provide relevant information related to the question.

        Context: {context}
        Question: {question}
        Answer: 
        """

        self.prompt = PromptTemplate(
            template=self.template, 
            input_variables=["context", "question"]
        )

        # Define the retriever with optimized search parameters
        self.retriever = self.db.as_retriever(search_kwargs={"k": 3, "fetch_k": 5})
        
        # Create a more efficient RAG chain with caching
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
if __name__ == "__main__":
    bot = LawBot()
    user_input = input("Ask me anything about the constitution: ")
    result = bot.get_response(user_input)
    print(result)





    






