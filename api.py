from flask import Flask, request, jsonify
import os
import pdfplumber
import docx
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

app = Flask(__name__)

# Define a placeholder for your OpenAI API key (replace with your actual key)
openai_api_key = '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'

# Define the file paths of the files in your local directory
# Replace these paths with the actual paths of your files
file_paths = ["cv.pdf"]

# Initialize conversation chain globally
conversation_chain = None

def read_files(file_paths):
    text = ""
    for file_path in file_paths:
        file_extension = os.path.splitext(file_path)[1]
        if file_extension == ".pdf":
            text += get_pdf_text(file_path)
        elif file_extension == ".docx":
            text += get_docx_text(file_path)
        else:
            text += get_csv_text(file_path)
    return text

def get_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def get_docx_text(docx_path):
    doc = docx.Document(docx_path)
    all_text = []
    for doc_para in doc.paragraphs:
        all_text.append(doc_para.text)
    text = ' '.join(all_text)
    return text

def get_csv_text(csv_path):
    # Placeholder for CSV processing logic
    return "CSV content goes here"

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    knowledge_base = FAISS.from_texts(text_chunks, embeddings)
    return knowledge_base

def initialize_conversation_chain():
    global conversation_chain
    text_chunks = get_text_chunks(read_files(file_paths))
    vector_store = get_vectorstore(text_chunks)
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

@app.route('/process_files', methods=['POST'])

def process_files():
    global conversation_chain
    if conversation_chain is None:
        initialize_conversation_chain()

    return jsonify({"message": "Files processed successfully."})

@app.route('/ask_question', methods=['POST'])
def ask_question():
    user_question = request.form.get("question")

    if not user_question:
        return jsonify({"error": "Please provide a question."}), 400

    if conversation_chain is None:
        return jsonify({"error": "Conversation chain not initialized. Please process files first."}), 400

    response = conversation_chain({'question': user_question})
    chat_history = response['chat_history']
    response_message = chat_history[-1].content

    return jsonify({"response": response_message})

def handle_user_input(user_question):
    with conversation_chain:
        response = conversation_chain({'question': user_question})
    return response['chat_history'][-1].content

if __name__ == '__main__':
    app.run(debug=True)
