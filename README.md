# 📄 RAGify

## 📖 Overview

This project is an AI-powered document question answering system developed using **Streamlit**, **LangChain**, **Google Gemini**, and **ChromaDB**. It allows users to upload multiple PDF and Microsoft Word documents, retrieve relevant information using semantic search, and generate context-aware answers based only on the uploaded documents.

## ✨ Features

- Upload multiple PDF and DOCX documents
- Extract text from uploaded documents
- Split documents into semantic chunks
- Generate embeddings using Google Gemini
- Store document embeddings in ChromaDB
- Perform semantic similarity search
- Generate answers using Google Gemini
- Support multiple document retrieval
- Display source references
- Maintain conversation history
- Reset conversation and clear indexed documents

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
             (Recursive Splitter)
                          │
                          ▼
                Gemini Embeddings
                          │
                          ▼
                 Chroma Vector Store
                          │
             User Question Submitted
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
| Embedding Model | Gemini Embedding 001 |
| Vector Database | ChromaDB |
| Framework | LangChain |
| PDF Processing | PyMuPDF |
| Word Processing | python-docx |

## 📂 Project Structure

```text
project/
│
├── app.py
├── rag_core.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env
```

## ⚙️ Installation

Clone the repository.

```bash
git clone https://github.com/your-username/your-repository.git
```

Move to the project directory.

```bash
cd your-repository
```

Create a virtual environment.

```bash
python -m venv venv
```

Activate the virtual environment.

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

Install the required packages.

```bash
pip install -r requirements.txt
```

## 🔑 API Configuration

Create a `.env` file in the project directory.

```text
GOOGLE_API_KEY=YOUR_API_KEY
```

## ▶️ Running the Application

```bash
streamlit run app.py
```

## 💻 Usage

1. Launch the application.
2. Upload one or more PDF or DOCX documents.
3. Wait until the documents are indexed.
4. Enter a question related to the uploaded documents.
5. View the generated answer along with the source references.

## ❓ Sample Questions

- Summarize the uploaded document.
- Which document discusses AI?
- Compare the focus of both documents.
- What recommendations are provided?
- What technologies are mentioned?
- Which document mentions emergency exits?

## 🚀 Future Enhancements

- OCR support for scanned PDFs
- Support for PowerPoint and Excel documents
- Chat memory
- Confidence score for retrieved answers
- Cloud deployment
- User authentication

## 👨‍💻 Author

**Impana L**

Master of Science in Information Systems  
Pace University

## 📄 License

This project was developed for academic and learning purposes.