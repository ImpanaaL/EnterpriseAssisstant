import streamlit as st
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key from .env
load_dotenv()

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

st.title("Safety & Compliance Knowledge Assistant")

question = st.text_input("Ask a question:")

if st.button("Submit"):
    if question:
        response = llm.invoke(question)
        st.write(response.content)
    else:
        st.warning("Please enter a question.")