import streamlit as st
import os
from openai import OpenAI
from os import environ
#step2
import tempfile
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
#step3
from langchain_text_splitters import RecursiveCharacterTextSplitter
#step4
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
import shutil
#step6
from openai import OpenAI

client = OpenAI(
	api_key=os.environ["API_KEY"],
	base_url="https://api.ai.it.cornell.edu",
)


#step1: support multiple files
st.title("📝 File Q&A with OpenAI")
uploaded_file = st.file_uploader(
    "Upload an article", 
    type=("txt", "pdf"),
    accept_multiple_files=True
)

question = st.chat_input(
    "Ask something about the article",
    disabled=not uploaded_file,
)

#step2: load documents
def load_documents(files):
    docs = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for f in files:
            p = Path(tmpdir) / f.name
            with open(p, "wb") as out:
                out.write(f.getvalue())
            if f.type == "text/plain":
                docs.extend(TextLoader(str(p)).load())
            elif f.type == "application/pdf":
                docs.extend(PyPDFLoader(str(p)).load())
    return docs

#step3: chunking
def chunk_documents(docs, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

#step4: create embeddings & chroma vector db
OPENAI_API_KEY = environ.get("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(model="openai.text-embedding-3-small", api_key=OPENAI_API_KEY)
#fixed
PERSIST_DIR = os.path.join(tempfile.gettempdir(), "chroma_db")
os.makedirs(PERSIST_DIR, exist_ok=True)

@st.cache_resource
def build_index_cached(chunks, persist_dir=PERSIST_DIR):
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

def reset_index():
    if Path(PERSIST_DIR).exists():
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)
        Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)

def build_index(chunks):
    reset_index()  # simplest: rebuild when new uploads arrive
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

#step5: retriever & fetch top k questions
def retrieve_context(vectordb, question, k=5):
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": k})
    results = retriever.invoke(question)
    return results

def format_context_and_sources(docs):
    ctx = "\n\n---\n\n".join(d.page_content for d in docs)
    src_lines = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        src_lines.append(f"[{i}] {src}" + (f" (page {page})" if page is not None else ""))
    return ctx, "\n".join(src_lines)


#step6
SYSTEM = (
    "You are a helpful assistant for question answering.\n"
    "Use ONLY the provided context to answer concisely (<=3 sentences).\n"
    "If the answer is not in the context, say you don't know."
)

#real loading & chunking
vectordb = None
if uploaded_file:
    docs = load_documents(uploaded_file)
    chunks = chunk_documents(docs)
    vectordb = build_index_cached(chunks)
    st.success(f"✅ Indexed {len(uploaded_file)} file(s) successfully!")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask something about the article"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

    

if question and uploaded_file:

    # Append the user's question to the messages
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    # Retrieve top-k relevant chunks
    search_results = retrieve_context(vectordb, question, k=5)
    ctx_text, sources = format_context_and_sources(search_results)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="openai.gpt-4o",  # Change this to a valid model name
            #changed
            messages=[
                {"role": "system", "content": SYSTEM + f"\n\nContext:\n{ctx_text}"},
                *st.session_state.messages
            ],
            stream=True
        )
        response = st.write_stream(stream)

    # Append the assistant's response to the messages
    st.session_state.messages.append({"role": "assistant", "content": response})