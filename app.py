from __future__ import annotations

import ast
import hashlib
import json
from typing import Any

import streamlit as st

from rag_core import INDEX_VERSION, RAGPipeline


USER_AVATAR = "👤"
ASSISTANT_AVATAR = "🤖"


st.set_page_config(
    page_title="RAGify | Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        .stApp {
            background-color: #f7f8fc;
        }

        .block-container {
            max-width: 1250px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        footer {
            visibility: hidden;
        }

        .rag-title {
            color: #172033;
            font-size: 2.7rem;
            font-weight: 800;
            letter-spacing: -1px;
            margin-bottom: 0.15rem;
        }

        .rag-subtitle {
            color: #667085;
            font-size: 1.05rem;
            margin-bottom: 2rem;
        }

        .question-heading,
        .conversation-heading {
            color: #172033;
            font-size: 1.75rem;
            font-weight: 750;
        }

        .question-heading {
            margin-top: 0.5rem;
            margin-bottom: 0.8rem;
        }

        .conversation-heading {
            margin-top: 2rem;
            margin-bottom: 1rem;
        }

        div[data-testid="stForm"] {
            background-color: #ffffff;
            border: 1px solid #d8dce6;
            border-radius: 13px;
            padding: 1rem;
            box-shadow: 0 2px 8px rgba(16, 24, 40, 0.04);
        }

        div[data-testid="stTextArea"] textarea {
            min-height: 120px;
            border: 1px solid #eaecf0;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1rem;
            resize: vertical;
        }

        div[data-testid="stTextArea"] textarea:focus {
            border-color: #6567f1;
            box-shadow: 0 0 0 1px #6567f1;
        }

        div[data-testid="stFormSubmitButton"] button {
            width: 100%;
            height: 50px;
            border: none;
            border-radius: 10px;
            background: linear-gradient(
                90deg,
                #6163ed 0%,
                #6967ef 100%
            );
            color: white;
            font-size: 1rem;
            font-weight: 600;
        }

        div[data-testid="stFormSubmitButton"] button:hover {
            color: white;
            background: linear-gradient(
                90deg,
                #5052dc 0%,
                #5856df 100%
            );
            transform: translateY(-1px);
            box-shadow: 0 6px 14px rgba(97, 99, 237, 0.22);
        }

        div[data-testid="stChatMessage"] {
            background-color: #ffffff;
            border: 1px solid #e0e3eb;
            border-radius: 15px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 7px rgba(16, 24, 40, 0.03);
        }

        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] li {
            color: #202939;
            font-size: 1rem;
            line-height: 1.7;
        }

        div[data-testid="stChatMessage"]:has(
            div[data-testid="chatAvatarIcon-assistant"]
        ) {
            background: linear-gradient(
                135deg,
                #fbfaff 0%,
                #f4f5ff 55%,
                #f3f8ff 100%
            );
            border-color: #d9dcfa;
        }

        .empty-conversation {
            color: #8a91a0;
            background-color: #ffffff;
            border: 1px dashed #d7dbe5;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
        }

        section[data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e4e7ec;
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.5rem;
        }

        .sidebar-title {
            color: #172033;
            font-size: 1.25rem;
            font-weight: 750;
            margin-bottom: 1rem;
        }

        .sidebar-stat-label {
            color: #667085;
            font-size: 0.85rem;
            margin-bottom: 0.25rem;
        }

        .sidebar-stat-value {
            color: #172033;
            font-size: 1.7rem;
            font-weight: 700;
        }

        div[data-testid="stFileUploader"] {
            border: 1px dashed #aab2ff;
            border-radius: 12px;
            padding: 0.5rem;
        }

        div[data-testid="stExpander"] {
            background-color: #ffffff;
            border: 1px solid #dfe3eb;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_state() -> None:
    stored_version = st.session_state.get("index_version")

    if stored_version != INDEX_VERSION:
        st.session_state.clear()
        st.session_state.index_version = INDEX_VERSION

    if "pipeline" not in st.session_state:
        try:
            st.session_state.pipeline = RAGPipeline()
        except Exception as error:
            st.error(
                f"Application initialization failed: {error}"
            )
            st.stop()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "indexed_signature" not in st.session_state:
        st.session_state.indexed_signature = ""

    if "index_result" not in st.session_state:
        st.session_state.index_result = None

    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []


initialize_state()
pipeline = st.session_state.pipeline



def extract_text_content(content: Any) -> str:
    """Convert Gemini/LangChain message content into clean display text."""
    if content is None:
        return ""

    if isinstance(content, str):
        cleaned = content.strip()
        if not cleaned:
            return ""

        # Gemini/LangChain can sometimes return a Python/JSON string
        # representation of content blocks instead of the blocks themselves.
        if cleaned.startswith(("[", "{")):
            parsed_content = None

            try:
                parsed_content = json.loads(cleaned)
            except (json.JSONDecodeError, TypeError):
                try:
                    parsed_content = ast.literal_eval(cleaned)
                except (ValueError, SyntaxError):
                    parsed_content = None

            if parsed_content is not None and parsed_content != content:
                extracted = extract_text_content(parsed_content)
                if extracted:
                    return extracted

        return cleaned

    if isinstance(content, dict):
        text_value = content.get("text")
        if isinstance(text_value, str) and text_value.strip():
            return text_value.strip()

        content_value = content.get("content")
        if content_value is not None:
            return extract_text_content(content_value)

        return ""

    if isinstance(content, (list, tuple)):
        text_parts = [
            extract_text_content(item)
            for item in content
        ]
        return "\n\n".join(
            part for part in text_parts if part
        ).strip()

    text_attribute = getattr(content, "text", None)
    if isinstance(text_attribute, str) and text_attribute.strip():
        return text_attribute.strip()

    content_attribute = getattr(content, "content", None)
    if content_attribute is not None and content_attribute is not content:
        extracted = extract_text_content(content_attribute)
        if extracted:
            return extracted

    return str(content).strip()

def create_upload_signature(
    uploaded_files: list[Any],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(INDEX_VERSION.encode("utf-8"))

    for uploaded_file in sorted(
        uploaded_files,
        key=lambda item: item.name,
    ):
        file_bytes = uploaded_file.getvalue()

        hasher.update(
            uploaded_file.name.encode(
                "utf-8",
                errors="ignore",
            )
        )
        hasher.update(file_bytes)

    return hasher.hexdigest()


def reset_documents() -> None:
    """Clear the vector index and all UI state related to the current documents."""
    try:
        pipeline.vector_manager.clear()
    except Exception as error:
        st.warning(f"The document index could not be cleared completely: {error}")

    st.session_state.index_result = None
    st.session_state.indexed_signature = ""
    st.session_state.chat_history = []


def reset_conversation_state() -> None:
    """Clear only the displayed conversation, keeping indexed documents available."""
    st.session_state.chat_history = []


def render_source(source: Any, source_number: int) -> None:
    metadata = getattr(source, "metadata", {}) or {}
    source_name = metadata.get("source", "Unknown document")
    location = metadata.get("location", "Unknown location")
    page_content = extract_text_content(getattr(source, "page_content", ""))

    st.markdown(f"**Source {source_number}: {source_name} — {location}**")

    if page_content:
        st.markdown(page_content[:1000])
        if len(page_content) > 1000:
            st.caption("Source preview shortened.")
    else:
        st.caption("No source preview is available.")


def normalize_chat_history() -> None:
    """Convert chat history from older tuple format to the current dictionary format."""
    normalized: list[dict[str, Any]] = []

    for item in st.session_state.chat_history:
        if isinstance(item, dict):
            role = item.get("role", "assistant")
            content = extract_text_content(item.get("content", ""))
            sources = list(item.get("sources") or [])
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            role = str(item[0])
            content = extract_text_content(item[1])
            sources = []
        else:
            continue

        normalized.append({"role": role, "content": content, "sources": sources})

    st.session_state.chat_history = normalized


normalize_chat_history()


with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">Document Workspace</div>',
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Upload PDF or Word documents.",
    ) or []

    if uploaded_files:
        current_signature = create_upload_signature(uploaded_files)

        if current_signature != st.session_state.indexed_signature:
            try:
                with st.spinner("Reading and indexing documents..."):
                    result = pipeline.index_files(uploaded_files)

                st.session_state.index_result = result
                st.session_state.indexed_signature = current_signature
                st.session_state.chat_history = []
            except Exception as error:
                st.session_state.index_result = None
                st.error(f"Indexing failed: {error}")
    elif st.session_state.indexed_signature:
        reset_documents()

    result = st.session_state.index_result

    if result is not None:
        if result.total_chunks > 0:
            st.success("Documents indexed successfully")

            file_column, chunk_column = st.columns(2)
            with file_column:
                st.markdown('<div class="sidebar-stat-label">Files</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="sidebar-stat-value">{result.total_files}</div>', unsafe_allow_html=True)
            with chunk_column:
                st.markdown('<div class="sidebar-stat-label">Chunks</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="sidebar-stat-value">{result.total_chunks}</div>', unsafe_allow_html=True)

            st.markdown(
                '<div class="sidebar-stat-label" style="margin-top:1rem;">Document units</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="sidebar-stat-value">{result.total_pages}</div>',
                unsafe_allow_html=True,
            )

            if result.errors:
                with st.expander("Indexing messages", expanded=False):
                    for error in result.errors:
                        st.warning(error)

            with st.expander("Indexed content", expanded=False):
                chunks = pipeline.vector_manager.get_all_chunks()

                if not chunks:
                    st.caption("No indexed chunks are available.")

                for chunk_number, chunk in enumerate(chunks, start=1):
                    metadata = getattr(chunk, "metadata", {}) or {}
                    source = metadata.get("source", "Unknown")
                    location = metadata.get("location", "Unknown location")
                    page = metadata.get("page", "?")
                    chunk_index = metadata.get("chunk_index", "?")
                    index_version = metadata.get("index_version", "unknown")

                    st.markdown(f"**Chunk {chunk_number}**")
                    st.caption(
                        f"{source} · {location} · page {page} · "
                        f"chunk_index {chunk_index} · {index_version}"
                    )
                    st.markdown(extract_text_content(chunk.page_content))

                    if chunk_number < len(chunks):
                        st.divider()
        else:
            st.error("No readable text was found in the selected documents.")

    st.divider()

    clear_documents = st.button(
        "Clear indexed documents",
        use_container_width=True,
        disabled=not bool(st.session_state.indexed_signature),
        key="clear_documents_button",
    )

    reset_conversation = st.button(
        "Reset conversation",
        use_container_width=True,
        disabled=len(st.session_state.chat_history) == 0,
        key="reset_conversation_button",
    )

    if clear_documents:
        reset_documents()
        st.rerun()

    if reset_conversation:
        reset_conversation_state()
        st.rerun()


st.markdown('<div class="rag-title">RAGify</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="rag-subtitle">
        Ask questions across PDF and Word documents using semantic search.
    </div>
    """,
    unsafe_allow_html=True,
)

index_result = st.session_state.index_result

st.markdown('<div class="question-heading">Ask your documents</div>', unsafe_allow_html=True)

has_documents = index_result is not None and index_result.total_chunks > 0

with st.form("question_form", clear_on_submit=True):
    question = st.text_area(
        "Question",
        placeholder="Example: Compare the focus of both documents.",
        label_visibility="collapsed",
        height=125,
    )

    left_space, button_column, right_space = st.columns([1, 2, 1])
    with button_column:
        submitted = st.form_submit_button(
            "Ask question",
            use_container_width=True,
            disabled=not has_documents,
        )

if submitted:
    cleaned_question = question.strip()

    if not has_documents:
        st.warning("Upload and index at least one readable document first.")
    elif not cleaned_question:
        st.warning("Please enter a question before submitting.")
    else:
        try:
            with st.spinner("Generating answer..."):
                answer_result = pipeline.ask(cleaned_question)

            clean_answer = extract_text_content(answer_result.answer)
            if not clean_answer:
                clean_answer = "I could not generate a readable answer from the model response."

            st.session_state.chat_history.append(
                {"role": "user", "content": cleaned_question, "sources": []}
            )
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": clean_answer,
                    "sources": list(answer_result.sources or []),
                }
            )
            st.rerun()
        except Exception as error:
            st.error(f"Unable to generate an answer: {error}")


st.markdown('<div class="conversation-heading">Conversation</div>', unsafe_allow_html=True)

if st.session_state.chat_history:
    for message_index, message in enumerate(st.session_state.chat_history):
        role = message.get("role", "assistant")
        content = extract_text_content(message.get("content", ""))
        sources = list(message.get("sources") or [])
        avatar = USER_AVATAR if role == "user" else ASSISTANT_AVATAR

        with st.chat_message(role, avatar=avatar):
            st.markdown(content)

            if role == "assistant" and sources:
                with st.expander(
                    f"Sources ({len(sources)})",
                    expanded=False,
                ):
                    for source_number, source in enumerate(sources, start=1):
                        render_source(source, source_number)
                        if source_number < len(sources):
                            st.divider()
else:
    st.markdown(
        """
        <div class="empty-conversation">
            Ask a question to start a conversation about your documents.
        </div>
        """,
        unsafe_allow_html=True,
    )