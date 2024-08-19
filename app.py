import streamlit as st 

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os 
import streamlit as st
groq_api_key = os.getenv('GROQ_API_KEY')


def load_document(file_name):
    name , extentsion = os.path.splitext(file_name)

    if extentsion == '.pdf':
        print(f'loading file {file_name}')
        loader = PyPDFLoader(file_name)
    elif extentsion == '.docx':
        print(f'loading file {file_name}')
        loader = Docx2txtLoader(file_name)
    elif extentsion == ".txt":
        print(f'loading file {file_name}')
        loader = TextLoader(file_name)
    else :
        print('please upload correct file')
        return None
    data = loader.load()
    return data 

def chunk_data(data , chunk_size=256 , chunk_overlap = 200 ):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size,chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks 


def create_embeddings(chunks):
    embeddings = HuggingFaceEmbeddings()
    db = Chroma.from_documents(chunks , embeddings)
    return db


def ask_and_get_answer(db,q,k=5):
    llm = ChatGroq(groq_api_key=groq_api_key, model='llama-3.1-70b-versatile', temperature=1)
    retriever = db.as_retriever(search_type='similarity', search_kwargs={'k': k})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
    answer = chain.invoke(q)
    return answer


st.subheader("LLM Q&A Application")
with st.sidebar:
    groq_api_key = st.text_input("API key", type='password')
    if groq_api_key:
        os.environ['GROQ_API_KEY'] = groq_api_key
    uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])
    chunk_size = st.number_input('chunk_size:', min_value=100, max_value=1000, value=256)
    k = st.number_input('k:', min_value=1, max_value=5, value=5)
    add_data = st.button('Add Data')

    if uploaded_file:
        with st.spinner('Reading, chunking, and embedding the file...'):
            bytes_data = uploaded_file.read()
            file_name = os.path.join("./", uploaded_file.name)
            with open(file_name, "wb") as f:
                f.write(bytes_data)

            data = load_document(file_name)
            chunks = chunk_data(data, chunk_size=chunk_size)
            db = create_embeddings(chunks)

            st.session_state.vs = db

            st.success("File uploaded, chunked, and embedded successfully")


# Initialize answer as an empty string
answer = ""

q = st.text_input('Ask a question about your file')

if q:
    if 'vs' in st.session_state:
        db = st.session_state.vs
        with st.spinner('Getting answer...'):
            try:
                answer = ask_and_get_answer(db, q, k=k)
                st.text_area('LLM Answer', answer['result'])
            except Exception as e:
                st.error(f'An error occurred: {str(e)}')

st.divider()

if 'history' not in st.session_state:
    st.session_state.history = " "

value = f'Q: {q}\nA: {answer}'

st.session_state.history = f"{value}\n{'.' * 100}\n{st.session_state.history}"
st.text_area(label="chat_history", key="history", height=400)
