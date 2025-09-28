import streamlit as st
import os
from openai import OpenAI
from os import environ
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader

#UI
st.title("📄 RAG File Q&A with OpenAI")
st.markdown("Upload **.txt** or **.pdf** files and ask questions about their content!")

uploaded_files = st.file_uploader("Upload Documents", type=["txt", "pdf"], accept_multiple_files=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask something about the uploaded documents"}]

#OpenAI setup
OPENAI_API_KEY = environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("⚠️ OpenAI API Key is missing! Set 'OPENAI_API_KEY' in environment variables.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model="openai.text-embedding-3-small", api_key=OPENAI_API_KEY)

#ChromaDB directory
persist_directory = "./chroma_db"

#Load
def load_documents(files):
    """Loads multiple documents using TextLoader & PyPDFLoader."""
    docs = []
    save_directory = "./data/knowledge_base" 

    os.makedirs(save_directory, exist_ok=True)  

    for file in files:
        temp_filepath = os.path.join(save_directory, file.name)
        
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())

        if file.type == "text/plain":
            loader = TextLoader(temp_filepath)
        elif file.type == "application/pdf":
            loader = PyPDFLoader(temp_filepath)
        
        docs.extend(loader.load())
        os.remove(temp_filepath)

    return docs

#Chunk Text
def chunk_documents(docs):
    """Splits documents into chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    return splitter.split_documents(docs)

#Split
if uploaded_files:
    documents = load_documents(uploaded_files) 
    chunks = chunk_documents(documents) 

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    st.success(f"✅ {len(uploaded_files)} file(s) processed successfully!")

#Chat
question = st.chat_input("Ask something about the document(s)", disabled=not uploaded_files)

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if question and uploaded_files:
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    #Search&Retrieve
    search_results = vectordb.similarity_search(question, k=5)
    retrieved_text = "\n\n".join([doc.page_content for doc in search_results])

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="openai.gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"Use the following retrieved document content:\n\n{retrieved_text}"},
                *st.session_state.messages
            ],
            stream=True
        )
        response = st.write_stream(stream)

    #Store
    st.session_state.messages.append({"role": "assistant", "content": response})
