import streamlit as st
from streamlit.logger import get_logger
import openai
import os

logger = get_logger(__name__)

def clear_query_history():
    st.session_state.clear()

def is_open_ai_key_valid(openai_api_key) -> bool:

    if not openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar!")
        return False
    os.environ["OPENAI_API_KEY"] = openai_api_key
    try:
        openai.chat.completions.create(
            model="gpt-3.5-turbo",
            # api_key=
            messages=[{"role": "user", "content": "test"}],
        )
    except Exception as e:
        st.error(f"{e.__class__.__name__}: {e}")
        logger.error(f"{e.__class__.__name__}: {e}")
        return False

    return True