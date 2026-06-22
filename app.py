import os
import uuid
import re
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


load_dotenv()

st.set_page_config(page_title="Safety & Compliance Knowledge Assistant", layout="wide")
st.title("Safety & Compliance Knowledge Assistant")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "indexed_file_names" not in st.session_state:
    st.session_state.indexed_file_names = []

if "total_files" not in st.session_state:
    st.session_state.total_files = 0

if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0

if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0


def extract_pdf_documents(uploaded_files):
    documents = []
    total_pages = 0

    for uploaded_file in uploaded_files:
        pdf_reader = PdfReader(uploaded_file)
        total_pages += len(pdf_reader.pages)

        for page_number, page in enumerate(pdf_reader.pages, start=1):
            text = page.extract_text()

            if text and text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": uploaded_file.name,
                            "page": page_number
                        }
                    )
                )

    return documents, total_pages


def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=250,
        separators=["\n\n", "\n", ". ", ": ", " ", ""]
    )

    return splitter.split_documents(documents)


def create_vectorstore(chunks):
    persist_directory = f"chroma_db_{uuid.uuid4().hex}"

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )


def clean_words(text):
    stopwords = {
        "what", "is", "the", "a", "an", "of", "in", "on", "for", "to",
        "from", "and", "or", "with", "this", "that", "are", "was", "were"
    }

    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

    return [word for word in words if word not in stopwords and len(word) > 2]


def keyword_search(question, chunks, limit=4):
    question_words = clean_words(question)
    scored_chunks = []

    for chunk in chunks:
        text = chunk.page_content.lower()
        score = sum(1 for word in question_words if word in text)

        if "equal employment opportunity" in question.lower():
            if "equal employment opportunity" in text or "equal opportunity employers" in text:
                score += 10

        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    return [chunk for score, chunk in scored_chunks[:limit]]


def retrieve_docs(question, vectorstore, chunks):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 20
        }
    )

    semantic_docs = retriever.invoke(question)
    keyword_docs = keyword_search(question, chunks, limit=4)

    final_docs = []
    seen = set()

    for doc in keyword_docs + semantic_docs:
        key = f"{doc.metadata.get('source')}-{doc.metadata.get('page')}-{doc.page_content[:200]}"

        if key not in seen:
            final_docs.append(doc)
            seen.add(key)

    return final_docs[:4]


def fallback_answer(docs):
    best_doc = docs[0]
    text = best_doc.page_content.strip()
    source = best_doc.metadata.get("source")
    page = best_doc.metadata.get("page")

    return f"{text} ({source}, Page {page})"


def generate_answer(question, docs):
    if not docs:
        return "I could not find this information in the uploaded PDFs."

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}\n{doc.page_content}"
        for doc in docs
    )

    prompt = PromptTemplate.from_template(
        """
You are a Safety & Compliance Knowledge Assistant.

Use ONLY the PDF context below.

If the context contains relevant information, answer from it.
Do not say the information is not found when the context contains related text.

PDF Context:
{context}

Question:
{question}

Answer clearly in 2-4 sentences.
Include the source file name and page number.
"""
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )

    response = llm.invoke(prompt.format(context=context, question=question))
    answer = response.content.strip()

    if "could not find" in answer.lower() or "not found" in answer.lower():
        return fallback_answer(docs)

    return answer


with st.sidebar:
    st.header("Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload one or more safety/compliance PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    current_file_names = [file.name for file in uploaded_files] if uploaded_files else []

    if uploaded_files and current_file_names != st.session_state.indexed_file_names:
        with st.spinner("Auto-indexing uploaded PDFs..."):
            documents, total_pages = extract_pdf_documents(uploaded_files)

            if not documents:
                st.error("No readable text found in the uploaded PDFs.")
                st.session_state.vectorstore = None
                st.session_state.chunks = []
                st.session_state.indexed_file_names = []
            else:
                chunks = create_chunks(documents)
                vectorstore = create_vectorstore(chunks)

                st.session_state.vectorstore = vectorstore
                st.session_state.chunks = chunks
                st.session_state.indexed_file_names = current_file_names
                st.session_state.total_files = len(uploaded_files)
                st.session_state.total_pages = total_pages
                st.session_state.total_chunks = len(chunks)
                st.session_state.chat_history = []

    if st.session_state.vectorstore is not None:
        st.success("PDFs indexed successfully!")
        st.write(f"Total files: {st.session_state.total_files}")
        st.write(f"Total pages: {st.session_state.total_pages}")
        st.write(f"Total chunks: {st.session_state.total_chunks}")

        with st.expander("View chunks"):
            for i, chunk in enumerate(st.session_state.chunks, start=1):
                st.markdown(f"### Chunk {i}")
                st.write(f"File: {chunk.metadata.get('source')}")
                st.write(f"Page: {chunk.metadata.get('page')}")
                st.write(chunk.page_content[:1000])


st.subheader("Ask a question from the uploaded PDFs")

question = st.text_input("Ask a question:")

if st.button("Submit"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif st.session_state.vectorstore is None:
        st.warning("Please upload PDFs first.")
    else:
        with st.spinner("Searching PDFs and generating answer..."):
            docs = retrieve_docs(
                question,
                st.session_state.vectorstore,
                st.session_state.chunks
            )

            answer = generate_answer(question, docs)

            st.session_state.chat_history.append(
                {
                    "question": question,
                    "answer": answer,
                    "sources": docs
                }
            )


st.divider()
st.subheader("Chat History")

for chat in st.session_state.chat_history:
    st.markdown(f"**You:** {chat['question']}")
    st.markdown(f"**Assistant:** {chat['answer']}")

    with st.expander("Sources Used"):
        for i, doc in enumerate(chat["sources"], start=1):
            st.markdown(f"**Source {i}**")
            st.write(f"File: {doc.metadata.get('source')}")
            st.write(f"Page: {doc.metadata.get('page')}")
            st.write(doc.page_content[:1200])

    st.divider()