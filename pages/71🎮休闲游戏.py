import streamlit as st
from streamlit_drawable_canvas import st_canvas

from utils import init_page_header, init_session_state, get_avatar
from utils import is_cuda_available, clear_cuda_cache, clear_streamlit_cache
from utils import update_aigc_temp_freq

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

import numpy as np


title = "休闲游戏"
icon = "🎮"
init_page_header(title, icon)
init_session_state()

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__() #调用父类方法初始化函数为网络
        self.fc1 = nn.Linear(28*28,512)  #第一层28*28是由图片像元数决定的，512为经验决定的
        self.fc2 = nn.Linear(512,128)  #512->128
        self.fc3 = nn.Linear(128,32)  #128->32
        self.fc4 = nn.Linear(32,10)  #最后一层的输出值为10（因为10分类，输出必须为10）  

    #网络的正向传播（计算过程）
    def forward (self, x):  
        x = F.relu(self.fc1(x))# h1 = relu(xw1+b1)       
        x = F.relu(self.fc2(x)) # h2 = relu(h1w2+b2)        
        x = F.relu(self.fc3(x))# h3 = relu(h2w3+b3)        
        x = self.fc4(x)# h4 = h3w4+b4（最后一次不加激活函数，使用均方差来计算）
        return x


@st.cache_resource
def get_cnn_net():
    net = Net()
    net.load_state_dict(torch.load('models/mnist/model_150.pkl'))# 读取模型
    return net


if __name__ == '__main__':
    # with st.expander("你写我猜", expanded=True):
    #     with st.form("form1", border=False):
    #         st.info("请写一个 0 ~ 9 的数字，让我猜猜您写的是哪个！", icon="🚨")
    #         canvas1 = st_canvas(key="canvas_game1",
    #                 height=500,
    #                 width=500,
    #                 stroke_width=20,
    #                 stroke_color="#CCCCCC"
    #                 )
    #         cols = st.columns(3)
    #         with cols[0]:
    #             cols = st.columns(3)
    #             with cols[0]:
    #                 form1_submit = st.form_submit_button("猜猜看", type="primary")
    #             with cols[1]:
    #                 form1_succ = st.form_submit_button("猜对了", type="primary")
    #             with cols[2]:
    #                 form1_fail = st.form_submit_button("猜错了", type="secondary")
    #         if form1_submit:
    #             net = get_cnn_net()
    #             image_array = np.asarray(Image.fromarray(canvas1.image_data).resize((28,28)).convert('L'))
    #             image_tensor=torch.FloatTensor(image_array.reshape(1,784))
    #             image_tensor = image_tensor / 255
    #             image_tensor = (image_tensor - 0.1307) / 0.3081
    #             pred = net(image_tensor.xpu()).argmax(dim=1)
    #             st.write(f"您写的是：{pred.item()}")
    #             st.image(image_array)
    #             update_aigc_temp_freq(1)
    #         if form1_succ:
    #             st.balloons()
    #         if form1_fail:
    #             st.snow()
    with st.expander("手绘着色", expanded=True):
        cols = st.columns([0.2, 0.4, 0.4])
        with cols[0]:
            stroke_width = st.slider("画笔粗细", min_value=1, max_value=50, step=1, value=5)
            stroke_color = st.color_picker("画笔颜色", value="#CCCCCC")
            drawing_mode = st.selectbox("绘画类型", options=['freedraw', 'transform', 'line', 'rect', 'circle', 'point', 'polygon'])
            fill_color = st.color_picker("填充颜色", value="#CCCCCC")
            upload_image = st.toggle("上传线描图")
            background_image=Image.open("statics/images/game_image_01.png")
            if upload_image:
                image_upload = st.file_uploader("上传线描图", type=["png","jpg"])
                if image_upload:
                    background_image = Image.open(image_upload)
            else:
                image_select = st.selectbox("内置线描图", options=["image_01","image_02","image_03","image_04","image_05"])
                background_image=Image.open(f"statics/images/game_{image_select}.png")
        with cols[1]:
            canvas2 = st_canvas(key="canvas_game2",
                    background_image=background_image,
                    height=733,
                    width=517,
                    stroke_width=stroke_width,
                    stroke_color=stroke_color,
                    drawing_mode=drawing_mode,
                    fill_color=fill_color
                    )

        form2_submit = st.button("画好了", type="primary")
        if form2_submit:
            with cols[2]:
                color_image = Image.fromarray(canvas2.image_data).resize(background_image.size)
                color_image_array = np.asarray(color_image)
                background_image_array = np.asarray(background_image)
                blend_image_array = np.where(color_image_array > 0, color_image_array, background_image_array)
                st.image(blend_image_array, width=517)
                update_aigc_temp_freq(1)
