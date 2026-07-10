import html
from typing import Any

import streamlit as st

from rag_core import RAGPipeline


st.set_page_config(
    page_title="RAGify | Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        /* Main application */
        .stApp {
            background-color: #f7f8fc;
        }

        .block-container {
            max-width: 1250px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* Hide default Streamlit footer */
        footer {
            visibility: hidden;
        }

        /* Main heading */
        .rag-title {
            font-size: 2.7rem;
            font-weight: 800;
            color: #172033;
            margin-bottom: 0.15rem;
            letter-spacing: -1px;
        }

        .rag-subtitle {
            font-size: 1.05rem;
            color: #667085;
            margin-bottom: 2rem;
        }

        /* Metric cards */
        .metric-card {
            min-height: 135px;
            padding: 1.25rem 1.35rem;
            background-color: #ffffff;
            border: 1px solid #d9dde7;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(16, 24, 40, 0.03);
        }

        .metric-label {
            color: #7a8292;
            font-size: 0.95rem;
            margin-bottom: 1.6rem;
        }

        .metric-value {
            color: #14213d;
            font-size: 2rem;
            font-weight: 700;
            line-height: 1;
        }

        /* Question section */
        .question-heading {
            color: #172033;
            font-size: 1.75rem;
            font-weight: 750;
            margin-top: 1.4rem;
            margin-bottom: 0.8rem;
        }

        div[data-testid="stForm"] {
            background-color: #ffffff;
            border: 1px solid #d8dce6;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 2px 8px rgba(16, 24, 40, 0.03);
        }

        div[data-testid="stTextArea"] textarea {
            min-height: 120px;
            border-radius: 10px;
            border: 1px solid #eaecf0;
            font-size: 1rem;
            padding: 1rem;
            resize: vertical;
        }

        div[data-testid="stTextArea"] textarea:focus {
            border-color: #6567f1;
            box-shadow: 0 0 0 1px #6567f1;
        }

        div[data-testid="stFormSubmitButton"] button {
            width: 100%;
            height: 48px;
            border: none;
            border-radius: 9px;
            background: linear-gradient(
                90deg,
                #6163ed 0%,
                #6967ef 100%
            );
            color: white;
            font-weight: 600;
            font-size: 1rem;
        }

        div[data-testid="stFormSubmitButton"] button:hover {
            background: linear-gradient(
                90deg,
                #5052dc 0%,
                #5856df 100%
            );
            color: white;
        }

        /* Conversation */
        .conversation-heading {
            color: #172033;
            font-size: 1.75rem;
            font-weight: 750;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }

        .user-message {
            background-color: #eef0ff;
            border: 1px solid #dde0ff;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.75rem;
        }

        .assistant-message {
            background-color: #ffffff;
            border: 1px solid #e0e3eb;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 2px 8px rgba(16, 24, 40, 0.03);
        }

        .message-role {
            color: #6b7280;
            font-size: 0.79rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.45rem;
        }

        .message-content {
            color: #202939;
            font-size: 1rem;
            line-height: 1.65;
            white-space: pre-wrap;
        }

        .empty-conversation {
            color: #8a91a0;
            background-color: #ffffff;
            border: 1px dashed #d7dbe5;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
        }

        /* Sidebar */
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

        /* Expander styling */
        div[data-testid="stExpander"] {
            background-color: #ffffff;
            border: 1px solid #dfe3eb;
            border-radius: 10px;
        }

        /* Reduce unwanted sidebar spacing */
        section[data-testid="stSidebar"] hr {
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_state() -> None:
    """Initialize all required Streamlit session-state variables."""

    if "pipeline" not in st.session_state:
        try:
            st.session_state.pipeline = RAGPipeline()
        except Exception as error:
            st.error(f"Application initialization failed: {error}")
            st.stop()

    defaults: dict[str, Any] = {
        "chat_history": [],
        "indexed_files": [],
        "index_result": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def create_file_signature(uploaded_files: list[Any]) -> list[str]:
    """
    Create a lightweight identifier for the currently uploaded files.

    This prevents documents from being indexed again on every Streamlit rerun.
    """

    return sorted(
        [
            (
                f"{uploaded_file.name}"
                f"::{uploaded_file.size}"
                f"::{uploaded_file.type}"
            )
            for uploaded_file in uploaded_files
        ]
    )


def reset_documents() -> None:
    """Clear the vector database and all related state."""

    try:
        st.session_state.pipeline.vector_manager.clear()
    except Exception:
        pass

    st.session_state.index_result = None
    st.session_state.indexed_files = []
    st.session_state.chat_history = []


def escape_and_format(text: Any) -> str:
    """Safely format plain text for display inside an HTML message card."""

    safe_text = html.escape(str(text))
    return safe_text.replace("\n", "<br>")


def render_metric_card(label: str, value: int) -> None:
    """Render a dashboard metric card."""

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{html.escape(label)}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_source(source: Any, source_number: int) -> None:
    """Render one retrieved source chunk."""

    metadata = getattr(source, "metadata", {}) or {}

    source_name = metadata.get(
        "source",
        "Unknown document",
    )

    location = metadata.get(
        "location",
        "Unknown location",
    )

    page_content = getattr(
        source,
        "page_content",
        "",
    )

    st.markdown(
        f"**Source {source_number}: "
        f"{source_name} — {location}**"
    )

    if page_content:
        preview = page_content[:1000]
        st.write(preview)

        if len(page_content) > 1000:
            st.caption("Source preview shortened.")
    else:
        st.caption("No source preview is available.")


initialize_state()

pipeline = st.session_state.pipeline


with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">Document Workspace</div>',
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Upload one or more PDF or Word documents.",
    )

    uploaded_files = uploaded_files or []

    current_file_signature = create_file_signature(
        uploaded_files
    )

    files_changed = (
        current_file_signature
        != st.session_state.indexed_files
    )

    if uploaded_files and files_changed:
        try:
            with st.spinner(
                "Reading and indexing documents..."
            ):
                index_result = pipeline.index_files(
                    uploaded_files
                )

            st.session_state.index_result = index_result
            st.session_state.indexed_files = (
                current_file_signature
            )
            st.session_state.chat_history = []

        except Exception as error:
            st.session_state.index_result = None
            st.error(f"Indexing failed: {error}")

   
    if (
        not uploaded_files
        and st.session_state.indexed_files
    ):
        reset_documents()

    index_result = st.session_state.index_result

    if index_result is not None:
        if index_result.total_chunks > 0:
            st.success("Documents indexed successfully")

            stat_column_1, stat_column_2 = st.columns(2)

            with stat_column_1:
                st.markdown(
                    """
                    <div class="sidebar-stat-label">
                        Files
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div class="sidebar-stat-value">
                        {index_result.total_files}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with stat_column_2:
                st.markdown(
                    """
                    <div class="sidebar-stat-label">
                        Chunks
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div class="sidebar-stat-value">
                        {index_result.total_chunks}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                """
                <div class="sidebar-stat-label"
                     style="margin-top: 1rem;">
                    Document units
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="sidebar-stat-value">
                    {index_result.total_pages}
                </div>
                """,
                unsafe_allow_html=True,
            )

            skipped_files = getattr(
                index_result,
                "skipped_files",
                [],
            )

            indexing_errors = getattr(
                index_result,
                "errors",
                [],
            )

            if skipped_files:
                st.warning(
                    "Skipped files: "
                    + ", ".join(skipped_files)
                )

            if indexing_errors:
                with st.expander("Indexing messages"):
                    for indexing_error in indexing_errors:
                        st.warning(indexing_error)

            with st.expander(
                "Indexed content",
                expanded=False,
            ):
                try:
                    chunks = (
                        pipeline.vector_manager
                        .get_all_chunks()
                    )
                except Exception as error:
                    chunks = []
                    st.warning(
                        f"Unable to load indexed chunks: "
                        f"{error}"
                    )

                if not chunks:
                    st.caption(
                        "No indexed chunks are available."
                    )

                for chunk_number, chunk in enumerate(
                    chunks,
                    start=1,
                ):
                    metadata = (
                        getattr(chunk, "metadata", {})
                        or {}
                    )

                    source_name = metadata.get(
                        "source",
                        "Unknown document",
                    )

                    location = metadata.get(
                        "location",
                        "Unknown location",
                    )

                    page_content = getattr(
                        chunk,
                        "page_content",
                        "",
                    )

                    st.markdown(
                        f"**Chunk {chunk_number}**"
                    )

                    st.caption(
                        f"{source_name} · {location}"
                    )

                    st.write(page_content[:500])

                    if len(page_content) > 500:
                        st.caption("Preview shortened.")

                    if chunk_number < len(chunks):
                        st.divider()

        else:
            st.error(
                "No readable text was found in the "
                "selected documents."
            )

            for indexing_error in getattr(
                index_result,
                "errors",
                [],
            ):
                st.warning(indexing_error)

    st.divider()

    clear_documents = st.button(
        "Clear indexed documents",
        use_container_width=True,
        disabled=not bool(
            st.session_state.indexed_files
        ),
    )

    reset_conversation = st.button(
        "Reset conversation",
        use_container_width=True,
        disabled=not bool(
            st.session_state.chat_history
        ),
    )

    if clear_documents:
        reset_documents()
        st.rerun()

    if reset_conversation:
        st.session_state.chat_history = []
        st.rerun()


st.markdown(
    '<div class="rag-title">RAGify</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="rag-subtitle">
        Ask questions across PDF and Word documents
        using semantic search.
    </div>
    """,
    unsafe_allow_html=True,
)

index_result = st.session_state.index_result

if index_result is not None:
    uploaded_file_count = index_result.total_files
    document_unit_count = index_result.total_pages
    indexed_chunk_count = index_result.total_chunks
else:
    uploaded_file_count = 0
    document_unit_count = 0
    indexed_chunk_count = 0

metric_column_1, metric_column_2, metric_column_3 = (
    st.columns(3, gap="large")
)

with metric_column_1:
    render_metric_card(
        "Uploaded files",
        uploaded_file_count,
    )

with metric_column_2:
    render_metric_card(
        "Document units",
        document_unit_count,
    )

with metric_column_3:
    render_metric_card(
        "Indexed chunks",
        indexed_chunk_count,
    )

st.markdown(
    '<div class="question-heading">Ask your documents</div>',
    unsafe_allow_html=True,
)

has_documents = (
    index_result is not None
    and index_result.total_chunks > 0
)

with st.form(
    "question_form",
    clear_on_submit=True,
):
    question = st.text_area(
        "Question",
        placeholder=(
            "Example: Compare the focus of both documents."
        ),
        label_visibility="collapsed",
        height=125,
    )

    submit_question = st.form_submit_button(
        "Generate answer",
        type="primary",
        use_container_width=True,
    )


if submit_question:
    cleaned_question = question.strip()

    if not cleaned_question:
        st.warning("Please enter a question.")

    elif not has_documents:
        st.warning(
            "Please upload and index at least one "
            "readable document."
        )

    else:
        try:
            with st.spinner(
                "Searching documents and generating answer..."
            ):
                answer_result = pipeline.ask(
                    cleaned_question
                )

            st.session_state.chat_history.append(
                {
                    "question": cleaned_question,
                    "answer": answer_result.answer,
                    "sources": answer_result.sources,
                }
            )

            st.rerun()

        except Exception as error:
            st.error(
                "Unable to answer the question: "
                f"{error}"
            )


st.markdown(
    '<div class="conversation-heading">Conversation</div>',
    unsafe_allow_html=True,
)

if not st.session_state.chat_history:
    st.markdown(
        """
        <div class="empty-conversation">
            Upload documents and ask a question to begin.
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    for conversation_number, chat in enumerate(
        reversed(st.session_state.chat_history),
        start=1,
    ):
        formatted_question = escape_and_format(
            chat.get("question", "")
        )

        formatted_answer = escape_and_format(
            chat.get("answer", "")
        )

        st.markdown(
            f"""
            <div class="user-message">
                <div class="message-role">You</div>
                <div class="message-content">
                    {formatted_question}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="assistant-message">
                <div class="message-role">
                    RAGify Assistant
                </div>
                <div class="message-content">
                    {formatted_answer}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sources = chat.get("sources", []) or []

        if sources:
            with st.expander(
                f"Sources used ({len(sources)})"
            ):
                for source_number, source in enumerate(
                    sources,
                    start=1,
                ):
                    render_source(
                        source,
                        source_number,
                    )

                    if source_number < len(sources):
                        st.divider()

        if conversation_number < len(
            st.session_state.chat_history
        ):
            st.write("")