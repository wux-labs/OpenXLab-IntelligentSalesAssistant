import streamlit as st
import streamlit_antd_components as sac

import os
import pandas as pd
from PIL import Image

from datetime import datetime

import torch
from database.database import engine
from sqlalchemy import text

from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document

from common.chat import load_model_by_id, combine_history
from common.chat import default_model
from common.product import image_chat_answer, load_huggingface_embedding, product_vector_index, save_product_ratings

from utils import init_page_header, init_session_state, get_avatar
from utils import is_cuda_available, is_cuda_enough, clear_cuda_cache, clear_streamlit_cache
from utils import cuda_size_24gb, cuda_size_40gb
from utils import image_to_base64
from utils import global_system_prompt


title = "商品管理"
icon = "🛍️"
init_page_header(title, icon)
init_session_state()

if "product_page" not in st.session_state.keys():
    st.session_state.product_page = 1

style_options = ["经典","淑女","浪漫","休闲","民族","学院","通勤","韩版","欧美"]
season_options = ["春季","夏季","秋季","冬季"]
user_text = "有一件{product_name}\n\n详细信息包括：\n{product_info}\n\n请你帮我列出它的6个亮点，每个亮点仅保留精简的4个字，用python list的形式输出：[特点1, 特点2, ...]，仅给出python list，不要输出其他内容，不要输出变量名称，不要输出警告"

page_size = 5
product_index_directory = "products/product_index"

def save_product_info(id, name, title, tags, image, video, images, gender, season, price, style, material, advantage, marketing, description):
    try:
        # if not advantage:
        #     tokenizer, model, deploy = load_model_by_id(default_model)
        #     messages = [
        #         {"role": "system", "content": global_system_prompt}
        #     ]
        #     if deploy == "huggingface":
        #         response, history = model.chat(
        #             tokenizer,
        #             combine_history(messages, user_text.format(product_name = name, product_info = description))
        #         )
        #     elif deploy == "lmdeploy":
        #         response = model.chat(
        #             combine_history(messages, user_text.format(product_name = name, product_info = description))
        #         ).response.text
        #     advantage = response
        with engine.connect() as conn:
            if id == "":
                sql = text(f'''
                insert into ai_labs_product_info(name, title, tags, image, video, images, gender, season, price, style, material, advantage, marketing, description, created_at, updated_at)
                values(:name, :title, :tags, :image, :video, :images, :gender, :season, :price, :style, :material, :advantage, :marketing, :description, :created_at, :updated_at)
                ''')
                conn.execute(sql, [{
                    'name': name,
                    'title': title,
                    'tags': tags,
                    'image': image,
                    'video': video,
                    'images': images,
                    'gender': gender,
                    'season': season,
                    'price': price,
                    'style': style,
                    'material': material,
                    'advantage': advantage,
                    'description': description,
                    'marketing': marketing,
                    'created_at': datetime.now(),
                    'updated_at': datetime.now()
                }])
            else:
                sql = text(f'''
                update ai_labs_product_info set name = :name, title = :title, tags = :tags, gender = :gender, season = :season, price = :price, style = :style, material = :material, advantage = :advantage, marketing = :marketing, description = :description, updated_at = :updated_at
                where id = :id
                ''')
                conn.execute(sql, [{
                    'name': name,
                    'title': title,
                    'tags': tags,
                    'gender': gender,
                    'season': season,
                    'price': price,
                    'style': style,
                    'material': material,
                    'advantage': advantage,
                    'marketing': marketing,
                    'description': description,
                    'updated_at': datetime.now(),
                    'id': id
                }])
            conn.commit()

            
            data = pd.read_sql(f"select * from ai_labs_product_info order by updated_at desc", conn)

            records = data.to_dict('records')
            docs = [Document(page_content=record["marketing"] + record["description"]+ datetime.now().strftime('%Y-%m-%d-%H%M%S'), 
                             metadata={"id": str(record["id"]), "name": record["name"], "title":record["title"], 
                                       "image":record["image"], "video":record["video"],
                                       "gender":record["gender"], "season":record["season"], "price":record["price"],
                                       "style":record["style"], "material":record["material"], "advantage":record["advantage"],
                                       "page": "1", "chunk": "0", "source": "1-0"
                                       })
                    for record in records]
            ids = [record["id"] for record in records]

            embedding = load_huggingface_embedding()
            vector_index = FAISS.from_texts([""], embedding)

            # vector_index = product_vector_index(product_index_directory)
            # for id in ids:
            #     try:
            #         vector_index.delete([id])
            #     except:
            #         pass
            vector_index.add_documents(docs, ids=ids)
            vector_index.save_local(product_index_directory)

            st.toast("保存成功！", icon="✌️")

    except Exception as e:
        st.exception(e)


def query_product_page_condition():
    with engine.connect() as conn:
        sql = f"""
            select id as 商品编号, name as 商品名称, title as 商品标题, tags as 商品标签,
                   gender as 商品类型, season as 适合季节, price as 商品价格,
                   style as 设计风格, material as 服装材质, advantage as 服装亮点,
                   marketing as 营销文案, description as 商品描述,
                   image as 商品主图, video as 商品视频
              from ai_labs_product_info
             where name like '{query_product_name}%'
               and gender like '{query_product_gender}%'
               and season like '{query_product_season}%'
               and price between {query_product_price[0]} and {query_product_price[1]}
             order by updated_at desc limit 100
            """
        records = pd.read_sql(sql, conn)

    tabs = st.tabs(["商品列表"])
    with tabs[0]:
        page_data = records[(int(st.session_state.product_page) - 1) * page_size: (int(st.session_state.product_page) * page_size)]
        page_data["商品主图"]=page_data["商品主图"].apply(lambda x: "data:image/png;base64," + image_to_base64(x))
        st.data_editor(
            pd.concat([page_data.iloc[:, -2], page_data.iloc[:, 1:-2]], axis=1),
            column_config={
                "商品主图": st.column_config.ImageColumn("商品主图")
            },
            hide_index=True,
            use_container_width=True,
            disabled=True,
        )
        sac.pagination(total=len(records), page_size=page_size, show_total=True, key='product_page')

        for _, row in page_data.iterrows():
            with st.expander(row.iloc[2]):
                cols = st.columns(3)
                with cols[0]:
                    tabs = st.tabs(["商品主图", "商品视频"])
                    with tabs[0]:
                        if row.iloc[12]:
                            st.image(row.iloc[12])
                    with tabs[1]:
                        if row.iloc[13]:
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
                if st.session_state.rolename == "卖家":
                    cols = st.columns([0.2,0.2,0.2,0.2,0.2])
                    with cols[0]:
                        if st.button("👚在线试穿", key=f"try_on_{row.iloc[0]}"):
                            st.session_state.tryon_id = row.iloc[0]
                            st.switch_page("pages/61👚在线试穿.py")
                    with cols[1]:
                        with st.popover("📝修改信息"):
                            with st.form(f"edit_product_{row.iloc[0]}_form"):
                                columns = st.columns(3)
                                with columns[0]:
                                    product_name = st.text_input(label="商品名称", key=f"product_name_{row.iloc[0]}", value=row.iloc[1])
                                    product_gender = st.selectbox("商品类型", key=f"product_gender_{row.iloc[0]}", options=["男装","女装"], index=["男装","女装"].index(row.iloc[4]))
                                    product_material = st.text_input(label="服装材质", key=f"product_material_{row.iloc[0]}", value=row.iloc[8])
                                with columns[1]:
                                    product_title = st.text_input(label="商品标题", key=f"product_title_{row.iloc[0]}", value=row.iloc[2])
                                    product_season = st.selectbox("适合季节", key=f"product_season_{row.iloc[0]}", options=season_options, index=season_options.index(row.iloc[5]))
                                    product_advantage = st.text_input(label="服装亮点", key=f"product_advantage_{row.iloc[0]}", value=row.iloc[9])
                                with columns[2]:
                                    product_tags = st.text_input(label="商品标签", key=f"product_tags_{row.iloc[0]}", value=row.iloc[3])
                                    product_style = st.selectbox(label="设计风格", key=f"product_style_{row.iloc[0]}", options=style_options, index=style_options.index(row.iloc[7]))
                                    product_price = st.slider("商品价格", key=f"product_price_{row.iloc[0]}", min_value=0.0, max_value=1000.0, step=0.1, value=row.iloc[6])
                                product_marketing = st.text_area(label="营销文案", key=f"product_marketing_{row.iloc[0]}", value=row.iloc[10])
                                product_description = st.text_area(label="商品描述", key=f"product_description_{row.iloc[0]}", value=row.iloc[11])
                                if st.form_submit_button("提交", type="primary", use_container_width=True):
                                    save_product_info(row.iloc[0], product_name, product_title, product_tags, "", "", "", product_gender, product_season, product_price, product_style, product_material, product_advantage, product_marketing, product_description)
                    with cols[2]:
                        with st.popover("📝我要评价"):
                            with st.form(f"comment_product_{row.iloc[0]}_form"):
                                rating = st.number_input("评分", key=f"product_rating_{row.iloc[0]}", min_value=1, max_value=5)
                                comment = st.text_area(label="评论", key=f"product_comment_{row.iloc[0]}")
                                if st.form_submit_button("提交", type="primary", use_container_width=True):
                                    save_product_ratings(row.iloc[0], rating, comment)
                else:
                    cols = st.columns([0.2,0.2,0.2,0.2,0.2])
                    with cols[0]:
                        if st.button("👚在线试穿", key=f"try_on_{row.iloc[0]}"):
                            st.session_state.tryon_id = row.iloc[0]
                            st.switch_page("pages/61👚在线试穿.py")
                    with cols[1]:
                        if st.button("🙋🏻商品咨询", key=f"ask_product_{row.iloc[0]}"):
                            st.session_state.ask_product_id = row.iloc[0]
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


if __name__ == '__main__':

    clear_streamlit_cache(["xcomposer2_vl_tokenizer", "xcomposer2_vl_model"])

    if st.session_state.rolename == "卖家":
        with st.expander("录入商品"):
            with st.form("add_product_info", clear_on_submit=False):
                cols = st.columns(3)
                with cols[0]:
                    product_name = st.text_input(label="商品名称", key="input_product_name")
                with cols[1]:
                    product_title = st.text_input(label="商品标题", key="input_product_title")
                with cols[2]:
                    product_tags = st.text_input(label="商品标签", key="input_product_tags")
                cols = st.columns(3)
                with cols[0]:
                    product_gender = st.selectbox("商品类型",options=["男装","女装"], key="input_product_gender")
                with cols[1]:
                    product_season = st.selectbox("适合季节",options=season_options, key="input_product_season")
                with cols[2]:
                    product_price = st.slider("商品价格", key="input_product_price", min_value=0.0, max_value=1000.0, step=0.1, value=50.0)
                cols = st.columns(3)
                with cols[0]:
                    product_style = st.selectbox(label="设计风格", key="input_product_style", options=style_options)
                with cols[1]:
                    product_material = st.text_input(label="服装材质", key="input_product_material")
                with cols[2]:
                    product_advantage = st.text_input(label="服装亮点", key="input_product_advantage")
                cols = st.columns(3)
                with cols[0]:
                    product_image = st.file_uploader("平铺图",type=["png","jpg"])
                with cols[1]:
                    product_video = st.file_uploader("视频", type=["mp4", "avi"])
                with cols[2]:
                    product_images = st.file_uploader("其他图",type=["png","jpg"], accept_multiple_files=True)
                if is_cuda_enough(cuda_size_24gb):
                    cols = st.columns(3)
                    with cols[0]:
                        marketing = st.form_submit_button("生成营销文案", use_container_width=True)
                        if marketing:
                            st.session_state.input_product_marketing = image_chat_answer(Image.open(product_image).convert('RGB'))
                            clear_cuda_cache()
                product_marketing = st.text_area("营销文案", key="input_product_marketing")
                product_description = st.text_area("商品描述", key="input_product_description")

                cols = st.columns(3)
                with cols[1]:
                    submitted = st.form_submit_button("保存商品信息", type="primary", use_container_width=True)
                if submitted:
                    os.makedirs(f"users/{st.session_state.username}/products/main", exist_ok=True)
                    os.makedirs(f"users/{st.session_state.username}/products/video", exist_ok=True)
                    os.makedirs(f"users/{st.session_state.username}/products/other", exist_ok=True)
                    image_main_path = f"users/{st.session_state.username}/products/main/{datetime.now().strftime('%Y-%m-%d-%H%M%S')}-{product_image.name}"
                    Image.open(product_image).save(image_main_path)
                    if product_video:
                        video_path = f"users/{st.session_state.username}/products/video/{datetime.now().strftime('%Y-%m-%d-%H%M%S')}-{product_video.name}"
                        with open(video_path, 'wb') as f:
                            f.write(product_video.getbuffer())
                    else:
                        video_path = None
                    images_path = []
                    for image_file in product_images:
                        image_file_path = f"users/{st.session_state.username}/products/other/{datetime.now().strftime('%Y-%m-%d-%H%M%S')}-{image_file.name}"
                        Image.open(image_file).save(image_file_path)
                        images_path.append(image_file_path)
                    save_product_info("", product_name, product_title, product_tags, image_main_path, video_path, ",".join(images_path), product_gender, product_season, product_price, product_style, product_material, product_advantage, product_marketing, product_description)

    with st.expander("查询商品"):
        with st.form("query_product_info"):
            cols = st.columns(3)
            with cols[0]:
                query_product_name = st.text_input(label="商品名称")
            with cols[1]:
                query_product_title = st.text_input(label="商品标题")
            with cols[2]:
                query_product_tags = st.text_input(label="商品标签")
            cols = st.columns(3)
            with cols[0]:
                query_product_gender = st.selectbox("商品类型",options=["", "男装","女装"])
            with cols[1]:
                query_product_season = st.selectbox("适合季节",options=["", "春季","夏季","秋季","冬季"])
            with cols[2]:
                query_product_price = st.slider("商品价格", min_value=0.0, max_value=1000.0, step=0.1, value=(50.0, 500.0))
            cols = st.columns(3)
            with cols[1]:
                query_submitted = st.form_submit_button("查询", type="primary", use_container_width=True)

    if query_submitted:
        st.session_state.product_page = 1

    query_product_page_condition()
