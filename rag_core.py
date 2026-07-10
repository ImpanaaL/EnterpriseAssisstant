import io
import os
import uuid
from dataclasses import dataclass
from typing import List

import fitz
from docx import Document as WordDocument
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)


load_dotenv()


@dataclass
class IndexResult:
    total_files: int
    total_pages: int
    total_chunks: int
    skipped_files: List[str]
    errors: List[str]


@dataclass
class AnswerResult:
    answer: str
    sources: List[Document]
    used_fallback: bool


class VectorManager:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is missing. Add it to your .env file."
            )

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
        )

        self.collection_name = self._new_collection_name()
        self.vectorstore = self._create_vectorstore()

    def _new_collection_name(self) -> str:
        return f"document_rag_{uuid.uuid4().hex}"

    def _create_vectorstore(self) -> Chroma:
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )

    def add_documents(self, documents: List[Document]) -> None:
        if documents:
            self.vectorstore.add_documents(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 8,
    ) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=k)

    def get_all_chunks(self) -> List[Document]:
        data = self.vectorstore.get()

        stored_documents = data.get("documents", [])
        stored_metadatas = data.get("metadatas", [])

        chunks = []

        for text, metadata in zip(
            stored_documents,
            stored_metadatas,
        ):
            chunks.append(
                Document(
                    page_content=text,
                    metadata=metadata or {},
                )
            )

        return chunks

    def clear(self) -> None:
        try:
            self.vectorstore.delete_collection()
        except Exception:
            pass

        self.collection_name = self._new_collection_name()
        self.vectorstore = self._create_vectorstore()


class RAGPipeline:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is missing. Add it to your .env file."
            )

        self.vector_manager = VectorManager()

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            google_api_key=api_key,
            temperature=0.2,
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=150,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

    def _extract_pdf(
        self,
        uploaded_file,
    ) -> List[Document]:
        documents = []
        file_bytes = uploaded_file.getvalue()

        pdf_document = fitz.open(
            stream=file_bytes,
            filetype="pdf",
        )

        try:
            for page_index in range(len(pdf_document)):
                page = pdf_document[page_index]
                text = page.get_text("text").strip()

                if not text:
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": uploaded_file.name,
                            "location": f"Page {page_index + 1}",
                            "page": page_index + 1,
                            "file_type": "PDF",
                        },
                    )
                )
        finally:
            pdf_document.close()

        return documents

    def _extract_docx(
        self,
        uploaded_file,
    ) -> List[Document]:
        file_bytes = uploaded_file.getvalue()

        word_document = WordDocument(
            io.BytesIO(file_bytes)
        )

        content_parts = []

        for paragraph in word_document.paragraphs:
            text = paragraph.text.strip()

            if text:
                content_parts.append(text)

        for table_index, table in enumerate(
            word_document.tables,
            start=1,
        ):
            rows = []

            for row in table.rows:
                cell_values = [
                    cell.text.strip()
                    for cell in row.cells
                ]

                if any(cell_values):
                    rows.append(" | ".join(cell_values))

            if rows:
                content_parts.append(
                    f"Table {table_index}\n"
                    + "\n".join(rows)
                )

        full_text = "\n\n".join(content_parts).strip()

        if not full_text:
            return []

        return [
            Document(
                page_content=full_text,
                metadata={
                    "source": uploaded_file.name,
                    "location": "Word document",
                    "page": 1,
                    "file_type": "DOCX",
                },
            )
        ]

    def index_files(
        self,
        uploaded_files,
    ) -> IndexResult:
        self.vector_manager.clear()

        all_documents = []
        skipped_files = []
        errors = []
        total_document_units = 0

        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            extension = os.path.splitext(file_name)[1].lower()

            try:
                if extension == ".pdf":
                    documents = self._extract_pdf(
                        uploaded_file
                    )

                elif extension == ".docx":
                    documents = self._extract_docx(
                        uploaded_file
                    )

                else:
                    documents = []
                    skipped_files.append(file_name)
                    errors.append(
                        f"{file_name}: Unsupported file type."
                    )

                if not documents:
                    if file_name not in skipped_files:
                        skipped_files.append(file_name)

                    if not any(
                        error.startswith(f"{file_name}:")
                        for error in errors
                    ):
                        errors.append(
                            f"{file_name}: No readable text was found."
                        )

                    continue

                total_document_units += len(documents)
                all_documents.extend(documents)

            except Exception as error:
                skipped_files.append(file_name)
                errors.append(
                    f"{file_name}: {str(error)}"
                )

        chunks = self.splitter.split_documents(
            all_documents
        )

        if chunks:
            self.vector_manager.add_documents(chunks)

        return IndexResult(
            total_files=len(uploaded_files),
            total_pages=total_document_units,
            total_chunks=len(chunks),
            skipped_files=skipped_files,
            errors=errors,
        )

    def _retrieve_sources(
        self,
        question: str,
    ) -> List[Document]:
        retrieved = self.vector_manager.similarity_search(
            question,
            k=8,
        )

        all_chunks = self.vector_manager.get_all_chunks()

        available_sources = {
            chunk.metadata.get("source", "Unknown document")
            for chunk in all_chunks
        }

        represented_sources = {
            chunk.metadata.get("source", "Unknown document")
            for chunk in retrieved
        }

        for source_name in available_sources:
            if source_name in represented_sources:
                continue

            source_chunk = next(
                (
                    chunk
                    for chunk in all_chunks
                    if chunk.metadata.get(
                        "source",
                        "Unknown document",
                    )
                    == source_name
                ),
                None,
            )

            if source_chunk is not None:
                retrieved.append(source_chunk)

        unique_sources = []
        seen = set()

        for document in retrieved:
            key = (
                document.metadata.get("source"),
                document.metadata.get("location"),
                document.page_content,
            )

            if key not in seen:
                seen.add(key)
                unique_sources.append(document)

        return unique_sources[:10]

    def ask(
        self,
        question: str,
    ) -> AnswerResult:
        sources = self._retrieve_sources(question)

        if not sources:
            return AnswerResult(
                answer=(
                    "No relevant information was found "
                    "in the uploaded documents."
                ),
                sources=[],
                used_fallback=False,
            )

        context_parts = []

        for index, source in enumerate(
            sources,
            start=1,
        ):
            source_name = source.metadata.get(
                "source",
                "Unknown document",
            )

            location = source.metadata.get(
                "location",
                "Unknown location",
            )

            context_parts.append(
                f"[Source {index}: {source_name}, {location}]\n"
                f"{source.page_content}"
            )

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""
You are a document question-answering assistant.

Use only the uploaded document context provided below.

Instructions:
- Give a direct and complete answer.
- Do not invent information.
- Mention the relevant document name when useful.
- Use page or document-location information when available.
- When asked to compare documents, explain the focus of each document separately and then state the main difference.
- When asked about both documents, use information from both.
- If information is unavailable, state that it is not available in the uploaded documents.
- Do not discuss retrieval, chunks, embeddings, or the internal system.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        try:
            response = self.llm.invoke(prompt)
            answer = response.content

            if isinstance(answer, list):
                text_parts = []

                for item in answer:
                    if isinstance(item, dict):
                        item_text = item.get("text")

                        if item_text:
                            text_parts.append(str(item_text))
                    else:
                        text_parts.append(str(item))

                answer = " ".join(text_parts)

            answer = str(answer).strip()

            if not answer:
                answer = (
                    "The model returned an empty response. "
                    "Please try the question again."
                )

        except Exception as error:
            error_text = str(error)

            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
                or "quota" in error_text.lower()
            ):
                answer = (
                    "The Gemini API usage limit has been reached. "
                    "Please wait for the quota to reset."
                )

            elif (
                "404" in error_text
                or "NOT_FOUND" in error_text
            ):
                answer = (
                    "The configured Gemini model is unavailable. "
                    "Check the model name or list the models supported "
                    "by your API key."
                )

            else:
                answer = (
                    f"Error generating answer: {error_text}"
                )

        return AnswerResult(
            answer=answer,
            sources=sources,
            used_fallback=False,
        )