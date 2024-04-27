import streamlit as st
from streamlit_drawable_canvas import st_canvas

from utils import init_page_header, init_session_state, update_aigc_temp_freq

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

import numpy as np


title = "休闲游戏"
icon = "🎮"
init_page_header(title, icon)
init_session_state()
