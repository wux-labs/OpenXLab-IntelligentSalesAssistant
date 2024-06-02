import streamlit as st
import os
import pandas as pd

import torch, auto_gptq
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from auto_gptq.modeling import BaseGPTQForCausalLM
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from langchain.vectorstores.faiss import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings


from database.database import engine
from sqlalchemy import text
from datetime import datetime

from utils import is_cuda_available, is_cuda_enough
from utils import cuda_size_24gb, cuda_size_40gb
from utils import image_to_base64, update_aigc_perm_freq


marketing_prompt = """你是一位优秀的服装商品智能销售专家，你现在要推销一件服装，服装商品的详细信息如下：
商品名称：{product_name}
商品标签：{product_tags}
商品类型：{product_gender}
适合季节：{product_season}
商品价格：{product_price}
设计风格：{product_style}
服装材质：{product_material}
商品描述：{product_description}

请根据商品详情信息和图片中的商品信息，用中文，写一段5000个汉字的、适用于电商的营销文案，包含商品的颜色、款式、风格、适用场合等信息。请仅给出文案正文内容。

----------

文案格式要求如下：

【标题】
标题内容

【导语】
导语内容

【正文】
正文内容

【产品亮点】
产品亮点内容

【购买呼吁】
购买呼吁内容

【结尾】
结尾内容

----------

文案要点要求如下：

1、标题：请简洁明了，突出产品特点，创造好奇心，吸引用户，激发用户的兴趣。
2、导语：简介营销活动的背景，激发用户的兴趣。
3、正文：请结合商品详情信息、图片详情信息，详细介绍商品。通过讲述一个与产品有关的故事，在情感上与用户建立联系。
4、产品亮点：突出服装的质量、款式、面料、舒适度、设计独特性等优势。
5、购买呼吁：列出优惠信息，激活用户的购买欲望，促使用户下单。
6、结尾：突出可以为用户提供贴心的服务。

----------

文案样例：

【标题】
新品上市-你的衣柜还缺这件！

【导语】
亲爱的时尚达人们：
您是否厌倦了每天打开衣柜却找不到心仪的服装？您是否渴望在这个{product_season}季节里，让自己成为焦点，引领潮流？那么。您绝对不能错过我们的这款衣服。

【正文】
一位年轻女孩身着一袭鲜艳的红色旗袍，优雅地站在门廊前。她的眼神中流露出自信和优雅，让这袭红色的旗袍更加耀眼夺目。
这款旗袍采用了传统的斜襟设计，领口和袖口处精致的绣花图案点缀其中，凸显了手工匠人精湛的技艺和精湛的设计。这款旗袍选用了优质丝绸材质，手感细腻柔软，穿着舒适大方，尽显东方古典的优雅气质。
这样的红色旗袍，无论是搭配西式晚礼裙还是中式旗袍，都是一款非常百搭且时尚的服装。它不仅能彰显出独特的个性，还能凸显出女性的优美身姿和独特的气质。
这款红色旗袍，无论是作为礼服、宴会服装，还是作为时尚的服装，都能为您的穿搭增添一份优雅与独特。

【产品亮点】
这款产品的亮点包括：{product_advantage}。
另外，它还有以下一些亮点：
款式新颖：我们的服装采用{product_style}风格设计，紧跟全球时尚趋势，为您带来时尚界最前沿的设计。
材质优良：我们的服装精选优质面料{product_material}，舒适亲肤，让您穿出健康美丽。
价格实惠：我们的服装价格为{product_price}，在保证高品质的同时还拥有较低的价格，我们承诺提供最具性价比的选择。

【购买呼吁】
现在下单，享受专属优惠！不仅如此，您还可以获得神秘的小礼物哦！快来抢购吧，让这个季节的你与众不同！

【结尾】
有任何疑问或者需要帮助，请随时联系我们。我们的客服团队24小时在线，随时准备为您提供最贴心的服务。

"""

marketing_prompt2 = """你是一位优秀的服装智能销售专家，你现在要推销一件服装，服装的详细信息如下：
商品名称：{product_name}
商品标签：{product_tags}
商品类型：{product_gender}
适合季节：{product_season}
商品价格：{product_price}
设计风格：{product_style}
服装材质：{product_material}
商品描述：{product_description}

首先，请你发挥想象力，用一个小故事讲述图片中的信息。
其次，请你根据提供的商品信息结合图片中的内容，精确提取商品的亮点价值，放大亮点以激发用户的购买欲，用500字文案详细描述一下这件服装。
最后，请你牢记，所有内容请使用中文回答！内容必须基于商品信息撰写，禁止捏造内容！
"""

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


def load_internlm_xcomposer2_vl(**kwargs):
    if "xcomposer2_vl_model" not in st.session_state.keys():
        model_id_or_path = "models/internlm/internlm-xcomposer2-vl-7b"
        st.session_state["xcomposer2_vl_tokenizer"] = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
        st.session_state["xcomposer2_vl_model"] = AutoModelForCausalLM.from_pretrained(model_id_or_path, device_map='cuda', trust_remote_code=True).half().eval()
        st.session_state["xcomposer2_vl_model"].tokenizer = st.session_state["xcomposer2_vl_tokenizer"]
    return st.session_state["xcomposer2_vl_tokenizer"], st.session_state["xcomposer2_vl_model"]


def load_internlm_xcomposer2_vl_4bit(**kwargs):
    # model_id_or_path="models/internlm/internlm-xcomposer2-7b-4bit"
    if "xcomposer2_vl_model" not in st.session_state.keys():
        auto_gptq.modeling._base.SUPPORTED_MODELS = ["internlm"]
        torch.set_grad_enabled(False)
        model_id_or_path="models/internlm/internlm-xcomposer2-vl-7b-4bit"

        tokenizer_path = model_id_or_path

        st.session_state["xcomposer2_vl_model"] = InternLMXComposer2QForCausalLM.from_quantized(
            model_id_or_path,
            device_map="auto",
            trust_remote_code=True,
            offload_buffers=True,
            **kwargs
        ).eval()

        st.session_state["xcomposer2_vl_tokenizer"] = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )

        if is_cuda_available():
            st.session_state["xcomposer2_vl_model"] = st.session_state["xcomposer2_vl_model"].cuda()

    return st.session_state["xcomposer2_vl_tokenizer"], st.session_state["xcomposer2_vl_model"]


def image_chat_answer(image_pil):
    tokenizer, model = load_internlm_xcomposer2_vl() if is_cuda_enough(40950) else load_internlm_xcomposer2_vl_4bit()
    
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

    response, _ = model.chat(tokenizer, 
                            query="<ImageHere>", 
                            image=image, 
                            history=[(marketing_prompt.format(
                                product_name = st.session_state.input_product_name,
                                product_tags = st.session_state.input_product_tags,
                                product_gender = st.session_state.input_product_gender,
                                product_season = st.session_state.input_product_season,
                                product_price = st.session_state.input_product_price,
                                product_style = st.session_state.input_product_style,
                                product_material = st.session_state.input_product_material,
                                product_advantage = st.session_state.input_product_advantage,
                                product_description = st.session_state.input_product_description
                            ),"")], 
                            do_sample=True)
    return response


@st.cache_resource
def load_huggingface_embedding():
    embedding = HuggingFaceEmbeddings(model_name="models/GanymedeNil/text2vec-large-chinese")
    return embedding


def product_vector_index(persist_directory="products/product_index"):
    embedding = load_huggingface_embedding()
    if not os.path.exists(persist_directory):
        index = FAISS.from_texts([""], embedding)
    else:
        index = FAISS.load_local(persist_directory, embedding, allow_dangerous_deserialization=True)
    return index


def select_product(ids):
    with engine.connect() as conn:
        df = pd.read_sql(f"""
            select id as 商品编号, name as 商品名称, title as 商品标题, tags as 商品标签,
                   gender as 商品类型, season as 适合季节, price as 商品价格,
                   style as 设计风格, material as 服装材质, advantage as 服装亮点,
                   marketing as 营销文案, description as 商品描述,
                   image as 商品主图, video as 商品视频
              from ai_labs_product_info
             where id in ({ids})
            """, conn)
        return df


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


def display_products(ids):
    products = select_product(ids)
    products["商品主图"]=products["商品主图"].apply(lambda x: "data:image/png;base64," + image_to_base64(x))

    for _, row in products.iterrows():
        with st.expander(row.iloc[2]):
            cols = st.columns(3)
            with cols[0]:
                tabs = st.tabs(["商品主图", "商品视频"])
                with tabs[0]:
                    if row.iloc[12]:
                        st.image(row.iloc[12])
                with tabs[1]:
                    if row.iloc[10]:
                        st.video(row.iloc[13])
            with cols[1]:
                st.write("商品名称：" + row.iloc[1])
                st.write("商品标签：" + row.iloc[3])
                st.write("商品类型：" + row.iloc[4])
                st.write("适合季节：" + row.iloc[5])
                st.write("商品价格：" + str(row.iloc[6]))
                st.write("设计风格：" + row.iloc[7])
                st.write("服装材质：" + row.iloc[8])
                st.write("服装亮点：" + row.iloc[9])
            with cols[2]:
                st.markdown(row.iloc[11])
            st.markdown("-------")
            st.markdown(str(row.iloc[10]))
            st.markdown("-------")
            cols = st.columns([0.2,0.2,0.2,0.2,0.2])
            with cols[0]:
                if st.button("👚在线试穿",  key=f"try_on_{row.iloc[0]}"):
                    st.session_state.tryon_id = row.iloc[0]
                    st.switch_page("pages/61👚在线试穿.py")
            with cols[1]:
                if st.button("🙋🏻商品咨询", key=f"ask_product_{row.iloc[0]}"):
                    st.session_state.tryon_id = row.iloc[0]
                    st.switch_page("pages/42🙋🏻商品咨询.py")
            with cols[2]:
                if st.button("💰立即购买", key=f"buy_product_{row.iloc[0]}"):
                    update_aigc_perm_freq(int(row.iloc[6] * 10))
                    st.toast("购买成功，感谢您的支持！", icon="💰")
            with cols[3]:
                if st.button("🛒加购物车", key=f"favorite_product_{row.iloc[0]}"):
                    st.toast("加购成功，感谢您的支持！", icon="🛒")
            with cols[4]:
                with st.popover("📝我要评价"):
                    with st.form(f"comment_product_{row.iloc[0]}_form"):
                        rating = st.number_input("评分", key=f"product_rating_{row.iloc[0]}", min_value=1, max_value=5)
                        comment = st.text_area(label="评论", key=f"product_comment_{row.iloc[0]}")
                        if st.form_submit_button("提交", type="primary", use_container_width=True):
                            save_product_ratings(row.iloc[0], rating, comment)
