import streamlit as st
import pandas as pd
from openai import OpenAI

st.title("ChatGPT Clone")

uploaded_file = st.file_uploader("Upload your CSV")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.text("Uploaded Data Sample")
    st.write(df.head())
    st.text("Uploaded Data Stats")
    st.write(df.describe())

# initialize openAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What questions can I answer about this data?"):
    # Add dataset converted to a string to the prompt
    added_context = [
        "I have the following dataset.",
        df.to_string(),
        "I want you to help me answer questions regarding this data. When you answer, don't show your process, just give me the answer.",
        prompt,
    ]
    prompt_with_context = "\n".join(added_context)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt_with_context})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    print(len(st.session_state.messages))
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
