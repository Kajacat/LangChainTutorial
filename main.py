import streamlit as st
import langchain_helper as lch
import textwrap

st.title("Video Search and Answering System")

with st.sidebar:
    with st.form(key="form"):
        youtube_url = st.sidebar.text_area(label="Enter Youtube URL", max_chars=50)
        query = st.sidebar.text_area(
            label="Ask about this video", max_chars=50, key="query"
        )

        submit_button = st.form_submit_button(label="Submit")

if query and youtube_url:
    db = lch.create_vector_db_from_youtube(youtube_url)
    response = lch.get_response_from_query(db, query)
    st.subheader("Answer:")
    st.text(textwrap.fill(response, width=80))
