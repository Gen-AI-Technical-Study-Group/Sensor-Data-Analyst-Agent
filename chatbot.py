import streamlit as st
import pandas as pd
from openai import OpenAI
import os

from dotenv import load_dotenv

load_dotenv(".env")

st.title("ChatGPT Clone")

# initialize openAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

uploaded_file = st.file_uploader("Upload your CSV")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.text("Uploaded Data Sample")
    df = pd.read_csv(df.head())
    st.text("Uploaded Data Stats")
    st.write(df.describe())
