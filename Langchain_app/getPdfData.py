
from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader 

# Define a function to extract text from a list of documents.
def getTextFromDocuments(documents):
    text =''
    for doc in documents:
        text += doc.page_content
    return text

# Define a function to load a PDF file using the langchain library.
def langchainPyPDFLoader(fileName):
    pdf_loader = PyPDFLoader(fileName)
    documents = pdf_loader.load()
    mText = getTextFromDocuments(documents)
    return mText

# Define a function to read a PDF file using the PyPDF2 library.
def pyPdfReaderFun(fileName):
    reader = PdfReader(fileName) 
    text = ''
    for pg in range(len(reader.pages)):
        page = reader.pages[pg] 
        text += page.extract_text() 
    return text


