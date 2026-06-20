import streamlit as st
from dotenv import load_dotenv
import os
from pypdf import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

st.title("Safety & Compliance Knowledge Assistant")

uploaded_file = st.file_uploader("Upload your safety/compliance PDF", type=["pdf"])

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
    st.text_area("PDF Content", text, height=300)

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
            height=150
        )

st.divider()

st.subheader("Ask a general question")

question = st.text_input("Ask a question:")

if st.button("Submit"):
    if question:
        response = llm.invoke(question)
        st.write(response.content)
    else:
        st.warning("Please enter a question.")