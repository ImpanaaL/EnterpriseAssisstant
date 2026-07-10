# 📄 RAGify

## 📖 Overview

RAGify is an AI-powered document question-answering system built using Streamlit, LangChain, Google Gemini, and ChromaDB. It allows users to upload PDF and Microsoft Word documents, retrieve relevant information through semantic search, and generate answers based on the uploaded content.

## ✨ Features

- Upload multiple PDF and DOCX documents
- Extract text from uploaded files
- Split document content into chunks
- Generate document embeddings
- Store embeddings in ChromaDB
- Perform semantic similarity search
- Generate context-aware answers
- Retrieve information from multiple documents
- Display source references
- Maintain conversation history
- Clear indexed documents and reset conversations

## 🏗️ System Architecture

```text
Upload PDF / DOCX
        │
        ▼
Text Extraction
(PyMuPDF / python-docx)
        │
        ▼
Text Chunking
        │
        ▼
Gemini Embeddings
        │
        ▼
Chroma Vector Store
        │
        ▼
User Question
        │
        ▼
Similarity Search
        │
        ▼
Gemini Language Model
        │
        ▼
Answer with Source References
```

## 🛠️ Technologies Used

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Programming Language | Python |
| Language Model | Google Gemini |
| Embedding Model | Gemini Embedding |
| Vector Database | ChromaDB |
| Framework | LangChain |
| PDF Processing | PyMuPDF |
| Word Processing | python-docx |

## 📂 Project Structure

```text
project/
├── app.py
├── rag_core.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 💻 Usage

1. Launch the application.
2. Upload one or more PDF or DOCX documents.
3. Wait for the documents to be processed and indexed.
4. Enter a question related to the uploaded documents.
5. Review the generated answer and source references.

## 🚀 Future Enhancements

- OCR support for scanned PDFs
- PowerPoint and Excel document support
- Chat memory
- Answer confidence scores
- Cloud deployment
- User authentication

## 👨‍💻 Author

Impanaa L

## 📄 License

This project was developed for academic and learning purposes.