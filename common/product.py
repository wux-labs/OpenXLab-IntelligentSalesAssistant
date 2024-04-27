import streamlit as st
import os

import torch, auto_gptq
from transformers import AutoTokenizer
from auto_gptq.modeling import BaseGPTQForCausalLM
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from langchain.vectorstores.faiss import FAISS

from database.database import engine
from sqlalchemy import text
from datetime import datetime

from utils import is_cuda_available, load_huggingface_embedding


class InternLMXComposer2QForCausalLM(BaseGPTQForCausalLM):
    layers_block_name = "model.layers"
    outside_layer_modules = [
        'vit', 'vision_proj', 'model.tok_embeddings', 'model.norm', 'output', 
    ]
    inside_layer_modules = [
        ["attention.wqkv.linear"],
        ["attention.wo.linear"],
        ["feed_forward.w1.linear", "feed_forward.w3.linear"],
        ["feed_forward.w2.linear"],
    ]


@st.cache_resource
def load_internlm_xcomposer2(**kwargs):
    auto_gptq.modeling._base.SUPPORTED_MODELS = ["internlm"]
    torch.set_grad_enabled(False)

    # model_id_or_path="models/internlm/internlm-xcomposer2-7b-4bit"
    model_id_or_path="models/internlm/internlm-xcomposer2-vl-7b-4bit"
    tokenizer_path = model_id_or_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )
    model = InternLMXComposer2QForCausalLM.from_quantized(
        model_id_or_path,
        device_map="auto",
        trust_remote_code=True,
        offload_buffers=True,
        **kwargs
    )
    model = model.eval()

    if is_cuda_available():
        model = model.cuda()

    return tokenizer, model


def image_chat_answer(image_pil):
    tokenizer, model = load_internlm_xcomposer2()
    
    img_size = 500
    vis_processor = transforms.Compose([
            transforms.Resize((img_size, img_size),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    image = vis_processor(image_pil).unsqueeze(0).half()

    if is_cuda_available():
        image = image.cuda()

    response, _ = model.chat(tokenizer, query="<ImageHere>你是一位优秀的服装商品智能销售专家，请根据图片中的商品信息，写一段适用于电商的营销文案，包含商品的颜色、款式、风格、适用场合等信息，文案在300字左右。", image=image, history=[], do_sample=True)
    return response


@st.cache_resource
def product_vector_index():
    embedding = load_huggingface_embedding()
    persist_directory = f"users/{st.session_state.username}/products/product_index"
    if not os.path.exists(persist_directory):
        index = FAISS.from_texts([""], embedding)
        index.save_local(persist_directory)
    else:
        index = FAISS.load_local(persist_directory, embedding, allow_dangerous_deserialization=True)
    return index


def save_product_ratings(product_id, rating, comment):
    try:
        with engine.connect() as conn:
            date_time = datetime.now()
            sql = text(f'''
            insert into ai_labs_product_ratings(user_id, product_id, rating, comment, date_time)
            values(:user_id, :product_id, :rating, :comment, :date_time)
            ''')
            conn.execute(sql, [{
                'user_id': st.session_state.userid,
                'product_id': product_id,
                'rating': rating,
                'comment': comment,
                'date_time': date_time
            }])
            conn.commit()

            st.toast("评价成功！", icon="✌️")

    except Exception as e:
        st.exception(e)
