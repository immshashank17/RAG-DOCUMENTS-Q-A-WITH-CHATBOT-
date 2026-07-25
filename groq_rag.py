from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate   
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


load_dotenv()
model = ChatGroq(
    model="llama-3.1-8b-instant"
)


prompt = ChatPromptTemplate.from_template(
    '''Answer From the following context only
Please provide the most accurate results based on the context only

{context}

Question: {input}'''
)
def generate_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(
        model="nomic-embed-text")
        st.session_state.loader = PyPDFDirectoryLoader('data')
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        st.session_state.split_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(
        documents=st.session_state.split_docs,
        embedding=st.session_state.embeddings)
        st.write("Vector Database Created")
st.title("GROQ LPU RAG CHATBOT" )
st.write("please CLICK the BUTTON below to ADD Embeddings of your Personal Data")

if st.button("Generate Embeddings"):
    generate_embedding()
    
user_prompt=st.text_input("Enter your Query From the Uploaded Document")
if st.button("Answer"):
    if user_prompt:
        document_context=create_stuff_documents_chain(model,prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieved_data=create_retrieval_chain(retriever,document_context)
        response=retrieved_data.invoke({'input':user_prompt})
        
        st.write("Answer")
        
        st.write(response['answer'])
        
        with st.expander("Document Context"):
           if "context" in response:
            for i, doc in enumerate(response["context"]):
             st.write(f"### Chunk {i+1}")
             st.write(doc.page_content)
             st.write(doc.metadata)
             st.write("-----------")