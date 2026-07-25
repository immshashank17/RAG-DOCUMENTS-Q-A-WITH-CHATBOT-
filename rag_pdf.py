from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_classic.chains import (
    create_retrieval_chain,
    create_history_aware_retriever,
)

from langchain_classic.chains.combine_documents  import (
    create_stuff_documents_chain,
)

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()
st.title("Conversation RAG with PDF + Message History GROQ")

api_key = st.text_input(
    "Provide your GROQ API Key",
    type="password"
)

if api_key:

    model = ChatGroq(
        model_name="llama-3.1-8b-instant",
        groq_api_key=api_key
    )

    session_id = st.text_input(
        "Please Provide Your Session ID:",
        value="default-session"
    )

    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_file = st.file_uploader(
        "Upload Your PDFs:",
        type="pdf"
    )

    if uploaded_file:

        temp_pdf = "./temporary.pdf"

        with open(temp_pdf, "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = PyPDFLoader(temp_pdf)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200)

        splits = splitter.split_documents(documents)

        
        embeddings = OllamaEmbeddings( model="nomic-embed-text")

        vector = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db")

        retriever = vector.as_retriever()

        # Code for Storing History

        contextualize_system_prompt = (
        "Given a chat history and the latest user question, "
        "reformulate the question to make it standalone. "
        "Do not answer, only rephrase.")

        contextualize_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", contextualize_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
        ])
        history_aware_retriever = create_history_aware_retriever(model,retriever,contextualize_prompt)

        system_prompt = """
        You are a helpful AI assistant.
        For Question answering,
        Answer from the provided context only.
        If unsure, Say I Dont know
        context
       {context}"""

        qa_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
        ])

        document_chain = create_stuff_documents_chain( model, qa_prompt)

        retrieval_chain = create_retrieval_chain(history_aware_retriever,document_chain)
        def get_session_history(session_id):
            if session_id not in st.session_state.store:
               st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]


        conversational_rag_chain = RunnableWithMessageHistory(
            retrieval_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Ask a Question about Your PDF:")

        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke({"input": user_input},
            config={"configurable": {  "session_id": session_id  }} )

            st.subheader(" Assistant Answer:")
            st.write(response["answer"])

            with st.expander("Chat History"):
                st.write(session_history.messages)
else:
    st.warning("Please Provide Your GROQ API Key")