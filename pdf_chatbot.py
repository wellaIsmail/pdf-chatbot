import streamlit as st
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
import tempfile


# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

st.title("🧠 Chat with your PDF")

pdf = st.file_uploader("Upload a PDF", type="pdf")
if pdf is not None:
    # Save the uploaded PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf.read())
        tmp_file_path = tmp_file.name

    # Load the PDF using PyPDFLoader
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)

    query = st.text_input("Ask something about the PDF:")
    if query:
        docs = vectorstore.similarity_search(query)
        llm = OpenAI(temperature=0)
        chain = load_qa_chain(llm, chain_type="stuff")
        answer = chain.run(input_documents=docs, question=query)
        st.write("💬 Answer:", answer)

    # Optionally, delete the temporary file after processing
    os.remove(tmp_file_path)