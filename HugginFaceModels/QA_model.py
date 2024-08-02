# Use a pipeline as a high-level helper
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader



#file loader and preprocessor
def file_preprocessing(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()

    # Split the text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50, add_start_index=True)
    texts = text_splitter.split_text(text)

    final_text = ""
    for text in texts:
        final_text += text
    return final_text

def QA_pipeline(file_path):
    pipe_QA = pipeline(
        "question-answering",
        tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2"),
        model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
    )
    input_text = file_preprocessing(file_path)
    res = pipe_QA(input_text)
    return res


questionA = QA_pipeline("C://Users//Admin//Documents//gen_ai_training//pdfs//Lawbot_paper4-compressed.pdf")
