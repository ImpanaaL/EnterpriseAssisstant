import streamlit as st
from dotenv import load_dotenv
import os
from pypdf import PdfReader

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

st.title("Safety & Compliance Knowledge Assistant")

uploaded_file = st.file_uploader(
    "Upload your safety/compliance PDF",
    type=["pdf"]
)

retriever = None

if uploaded_file is not None:
    pdf_reader = PdfReader(uploaded_file)

    text = ""

    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    st.success("PDF uploaded successfully!")
    st.write(f"Total pages: {len(pdf_reader.pages)}")

    st.subheader("Extracted PDF Text")
    st.text_area("PDF Content", text, height=250)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = text_splitter.split_text(text)

    st.subheader("Chunking Result")
    st.write(f"Total chunks created: {len(chunks)}")

    for i, chunk in enumerate(chunks[:3]):
        st.write(f"Chunk {i + 1}")
        st.text_area(
            f"Preview Chunk {i + 1}",
            chunk,
            height=120
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 2}
    )

    st.success("PDF indexed successfully in ChromaDB!")

st.divider()

st.subheader("Ask a question from your PDF")

question = st.text_input("Ask a question:")

if st.button("Submit"):
    if question:
        if retriever is None:
            st.warning("Please upload a PDF first.")
        else:
            docs = retriever.invoke(question)

            context = ""

            for doc in docs:
                context += doc.page_content + "\n\n"

            prompt = f"""
Answer the question using only the context below.
If the answer is not found in the context, say:
"I could not find this information in the uploaded PDF."

Context:
{context}

Question:
{question}
"""

            response = llm.invoke(prompt)

            st.subheader("Answer")
            st.write(response.content)

            st.subheader("Retrieved Sources")
            for i, doc in enumerate(docs):
                st.write(f"Source {i + 1}")
                st.write(doc.page_content)

    else:
        st.warning("Please enter a question.")