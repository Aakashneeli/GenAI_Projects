#Getting text from pdf


from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
import fitz 

def getTextFromDocuments(pdf_file):
    text =''
    doc = fitz.open(pdf_file)
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        text += page.get_text()
    doc.close()
    return text 




pdf_file = "C://Users//Admin//Documents//gen_ai_training//pdfs//Sports_analytics.pdf"
text_content = getTextFromDocuments(pdf_file)
print(text_content)
