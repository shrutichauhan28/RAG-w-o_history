import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import openai
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Load the GROQ API Key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ API Key is missing. Please check your .env file.")
    
# Initialize the LLM with GROQ API
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Initialize prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

from langchain_huggingface import HuggingFaceEmbeddings

# Initialize HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Function to extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        # Save the uploaded file temporarily
        with open(pdf.name, "wb") as f:
            f.write(pdf.getbuffer())
        
        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(pdf.name)
        pages = loader.load()
        for page in pages:
            text += page.page_content
        
        # Clean up by deleting the temporary file
        os.remove(pdf.name)
        
    return text

# Function to create vector embeddings
def create_vector_embedding(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.create_documents(text)
    vectors = FAISS.from_documents(documents, embeddings)
    return vectors

# Streamlit application
st.title("RAG Document Q&A With Groq And Lama3")

user_prompt = st.text_input("Enter your query from the research paper")

# Slider to control the number of documents used
num_docs = st.slider("Select number of documents to use", min_value=1, max_value=50, value=10)

with st.sidebar:
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type="pdf")
    
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            st.session_state.vectors = create_vector_embedding(raw_text)
            st.balloons()
            st.success("Your file has been processed, you can ask questions now!")

if user_prompt:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        elapsed_time = time.process_time() - start
        st.write(f"Response time: {elapsed_time}")

        st.write(response['answer'])

        # Display document similarity search results in a Streamlit expander
        with st.expander("Document similarity Search"):
            for doc in response['context'][:num_docs]:
                st.write(doc.page_content)
                st.write('------------------------')
    else:
        st.error("Please initialize the vector database first by clicking 'Submit & Process'.")
