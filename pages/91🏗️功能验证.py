import streamlit as st
import torch
import gc
import os, subprocess

from utils import init_page_header, init_session_state
from PIL import Image


title = "功能验证"
icon = "🏗️"
init_page_header(title, icon)
init_session_state()

if st.button("清理缓存"):
    torch.cuda.empty_cache()

cmd_text = st.chat_input("您的输入...")

if cmd_text:
    st.info(subprocess.getoutput(cmd_text).replace("\n", "  \n"))
