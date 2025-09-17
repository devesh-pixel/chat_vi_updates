

import os
import streamlit as st
from vic import unified_answer
import sys


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

st.set_page_config(page_title="Investment Updates Chat", layout="centered")
st.title("Investment Updates Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask about a company or compare multiple…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            reply = unified_answer(prompt)
        except Exception as e:
            reply = f"Error: {e}"
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})