import streamlit as st

from rag_core import RAGPipeline


st.set_page_config(
    page_title="Document Q&A Assistant",
    layout="wide",
)

st.title("Document Q&A Assistant")
st.caption(
    "Upload PDF or Word documents and ask questions from them."
)


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

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "index_result" not in st.session_state:
    st.session_state.index_result = None


pipeline = st.session_state.pipeline


with st.sidebar:
    st.header("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or Word files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )

    current_files = []

    if uploaded_files:
        current_files = [
            f"{uploaded_file.name}-{uploaded_file.size}"
            for uploaded_file in uploaded_files
        ]

    files_changed = (
        current_files
        != st.session_state.indexed_files
    )

    if uploaded_files and files_changed:
        try:
            with st.spinner(
                "Reading and indexing documents..."
            ):
                result = pipeline.index_files(
                    uploaded_files
                )

            st.session_state.index_result = result
            st.session_state.indexed_files = current_files
            st.session_state.chat_history = []

        except Exception as error:
            st.error(
                f"Indexing failed: {error}"
            )

    if not uploaded_files and st.session_state.indexed_files:
        pipeline.vector_manager.clear()
        st.session_state.indexed_files = []
        st.session_state.index_result = None
        st.session_state.chat_history = []

    result = st.session_state.index_result

    if result is not None:
        if result.total_chunks > 0:
            st.success(
                "Documents indexed successfully"
            )

            st.write(
                f"Files selected: {result.total_files}"
            )

            st.write(
                f"Document units processed: "
                f"{result.total_pages}"
            )

            st.write(
                f"Chunks created: "
                f"{result.total_chunks}"
            )

            if result.skipped_files:
                st.warning(
                    "Skipped files: "
                    + ", ".join(result.skipped_files)
                )

            if result.errors:
                with st.expander("Indexing messages"):
                    for error in result.errors:
                        st.warning(error)

            with st.expander("View indexed chunks"):
                chunks = (
                    pipeline.vector_manager
                    .get_all_chunks()
                )

                if not chunks:
                    st.caption(
                        "No chunks are currently indexed."
                    )

                for index, chunk in enumerate(
                    chunks,
                    start=1,
                ):
                    source_name = chunk.metadata.get(
                        "source",
                        "Unknown document",
                    )

                    location = chunk.metadata.get(
                        "location",
                        "Unknown location",
                    )

                    st.markdown(
                        f"**Chunk {index}**"
                    )

                    st.caption(
                        f"{source_name} — {location}"
                    )

                    st.write(
                        chunk.page_content[:500]
                    )

                    if len(chunk.page_content) > 500:
                        st.caption(
                            "Preview shortened."
                        )

                    st.divider()

        else:
            st.error(
                "No readable text was found "
                "in the selected documents."
            )

            for error in result.errors:
                st.warning(error)

    st.divider()

    if st.button(
        "Clear indexed documents",
        use_container_width=True,
    ):
        pipeline.vector_manager.clear()

        st.session_state.index_result = None
        st.session_state.indexed_files = []
        st.session_state.chat_history = []

        st.rerun()

    if st.button(
        "Reset conversation",
        use_container_width=True,
    ):
        st.session_state.chat_history = []
        st.rerun()


st.subheader("Ask a question")

question = st.text_input(
    "Your question",
    placeholder=(
        "Example: Compare the focus "
        "of both documents."
    ),
)

submit = st.button(
    "Submit",
    type="primary",
)


has_documents = (
    st.session_state.index_result is not None
    and st.session_state.index_result.total_chunks > 0
)


if submit:
    cleaned_question = question.strip()

    if not cleaned_question:
        st.warning(
            "Please enter a question."
        )

    elif not has_documents:
        st.warning(
            "Please upload and index "
            "at least one readable document."
        )

    else:
        try:
            with st.spinner(
                "Searching documents..."
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
                f"Unable to answer the question: {error}"
            )


st.divider()
st.subheader("Conversation")

if not st.session_state.chat_history:
    st.caption(
        "No questions have been asked yet."
    )

for chat in reversed(
    st.session_state.chat_history
):
    st.markdown(
        f"**You:** {chat['question']}"
    )

    st.markdown(
        f"**Assistant:** {chat['answer']}"
    )

    sources = chat.get(
        "sources",
        [],
    )

    if sources:
        with st.expander(
            f"Sources used ({len(sources)})"
        ):
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

                st.markdown(
                    f"**Source {index}: "
                    f"{source_name}, {location}**"
                )

                st.write(
                    source.page_content[:800]
                )

                if len(source.page_content) > 800:
                    st.caption(
                        "Source preview shortened."
                    )

                st.divider()

    st.divider()