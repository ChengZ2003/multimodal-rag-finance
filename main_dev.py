import os
import re
import asyncio

import streamlit as st

from sidebar_dev import (
    sidebar,
    build_query_engine,
    get_milvus_collections_list,
    DATA_DIR,
)
from ui_dev import clear_query_history

logo_path = "/root/autodl-tmp/avatar.png"

st.header("multimodal-rag-finance demo")

def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


if "is_ready" not in st.session_state.keys():
    st.session_state["is_ready"] = False

get_milvus_collections_list()
sidebar()

build_query_engine()


if st.session_state["is_ready"]:
    current_doc_id = re.search(r"\d+", st.session_state["selected_doc"]).group()
    current_doc = f"{current_doc_id}.pdf"
    current_doc_path = os.path.join(DATA_DIR, current_doc_id)
    st.write("当前文档：", f"`{current_doc}`")
    st.markdown("---")
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "有什么能够帮到您？"}
        ]

    for message in st.session_state.messages:
        avatar = logo_path if message["role"] == "assistant" else "🧑‍💻"
        with st.chat_message(message["role"], avatar=avatar):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant", avatar=logo_path):
            with st.spinner("Thinking ... "):
                resp = st.session_state["query_engine"].query(prompt)
                response, sources = resp.response, resp.source_nodes

            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)

            st.markdown("-------------------")
            for idx in range(len(sources)):
                # st.write(f"源文档 {idx+1}:\n{sources[idx].text}")
                st.write(f"源文档 **{idx+1}**:")
                st.write(f"{sources[idx].text}")
                # st.write(f"相关得分: {sources[idx].score}")
                page_number = sources[idx].metadata.get("page_number", "1")
                st.write(f"源页码: **{page_number}**")
                st.markdown("-------------------")

            # sources = [sources[idx].text for idx in range(len(sources))]
            # st.write(sources)

else:
    clear_query_history()
