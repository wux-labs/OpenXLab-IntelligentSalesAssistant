import streamlit as st
from utils import init_page_header, init_session_state

title = "系统介绍"
icon = "📖"
init_page_header(title, icon)
init_session_state()


if __name__ == '__main__':
    tabs = st.tabs(["获客","活客","留客"])
    with tabs[0]:
        cols = st.columns(2)
        with cols[0]:
            st.image("statics/docs/image_01.png")
        with cols[1]:
            st.image("statics/docs/image_02.png")
        cols = st.columns(2)
        with cols[0]:
            st.image("statics/docs/image_03.png")
        with cols[1]:
            st.image("statics/docs/image_04.png")
    with tabs[1]:
        cols = st.columns(2)
        with cols[0]:
            st.image("statics/docs/image_05.png")
        with cols[1]:
            st.image("statics/docs/image_06.png")
        st.image("statics/docs/image_07.png")
