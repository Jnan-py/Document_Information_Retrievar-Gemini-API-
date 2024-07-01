import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """ 
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,
    search for the details within all the documents provided, not only one document, the question may be from more than one document combined so search through all of them, 
    if the answer is not in provided context then search for similar content in the context provided and give the answer, 
    don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt = PromptTemplate(template=prompt_template,input_variables = ["context","question"])
    chain = load_qa_chain(model,chain_type="stuff",prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question":user_question},
        return_only_outputs=True
    )

    return response["output_text"]

def main():
    st.title("Document Information Retriever")

    with st.sidebar:
        st.title("Document Information Retriever")
        container = st.container(border=True)
        container.subheader("Written By: Jnan Yalla")
        container.write("**Email:** jnanyalla@gmail.com")
        container.write("**Phone:** +91-9150859936")

    st.subheader("File Uploading")
    pdf_docs = st.file_uploader("Upload your PDF files",type=['pdf'],accept_multiple_files=True)
    try : 
        if st.button("Submit the files"):
            with st.spinner("Uploading....."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Uploaded")

    except Exception as e:
        st.warning("Upload the files before submitting")

    st.header("Retrieve data from pdfs")
    if 'messages' not in st.session_state:
        st.session_state.messages = []

        st.session_state.messages.append(
            {
                'role':'assistant',
                'content':'Welcome !! Start asking your queries once after uploading the documents'
            }
        )

    for message in st.session_state.messages:
        row = st.columns(2)
        if message['role']=='user':
            row[1].chat_message(message['role']).markdown(message['content'])
        else:
            row[0].chat_message(message['role']).markdown(message['content'])
    try:
        user_question = st.chat_input("Enter your query here !!")
    
        if user_question:
            row_u = st.columns(2)
            row_u[1].chat_message('user').markdown(user_question)
            st.session_state.messages.append(
                {'role':'user',
                'content':user_question}
            )

            resp = user_input(user_question)
            
            row_a = st.columns(2)
            row_a[0].chat_message('assistant').markdown(resp)
            st.session_state.messages.append(
                {'role':'assistant',
                'content':resp}
            )

    except Exception as e:
        st.chat_message('assistant').markdown('The files are not uploaded or there might be an error, try uploading your files and proceed')
        st.session_state.messages.append(
            {
                'role':'assistant',
                'content':'The files are not uploaded or there might be an error, try uploading your files and proceed'
            }
        )
if __name__ == "__main__":
    main()