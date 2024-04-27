import streamlit as st
import streamlit_antd_components as sac

import os
import pandas as pd
from PIL import Image

from datetime import datetime

import torch
from database.database import engine
from sqlalchemy import text

from langchain.docstore.document import Document

from common.product import image_chat_answer, product_vector_index, save_product_ratings

from utils import get_avatar, init_page_header, init_session_state
from utils import select_aigc_left_freq, update_aigc_perm_freq, use_limited
from utils import image_to_base64, load_huggingface_embedding


title = "商品管理"
icon = "🛍️"
init_page_header(title, icon)
init_session_state()

if "product_page" not in st.session_state.keys():
    st.session_state.product_page = 1

page_size = 5

def save_product_info(id, name, title, tags, image, video, images, gender, season, price, marketing, description):
    try:
        with engine.connect() as conn:
            if id == "":
                sql = text(f'''
                insert into ai_labs_product_info(name, title, tags, image, video, images, gender, season, price, marketing, description, created_at, updated_at)
                values(:name, :title, :tags, :image, :video, :images, :gender, :season, :price, :marketing, :description, :created_at, :updated_at)
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
                    'description': description,
                    'marketing': marketing,
                    'created_at': datetime.now(),
                    'updated_at': datetime.now()
                }])
            else:
                sql = text(f'''
                update ai_labs_product_info set name = :name, title = :title, tags = :tags, gender = :gender, season = :season, price = :price, description = :description, updated_at = :updated_at
                where id = :id
                ''')
                conn.execute(sql, [{
                    'name': name,
                    'title': title,
                    'tags': tags,
                    'gender': gender,
                    'season': season,
                    'price': price,
                    'description': description,
                    'updated_at': datetime.now(),
                    'id': id
                }])
            conn.commit()

            
            data = pd.read_sql(f"select * from ai_labs_product_info order by updated_at desc limit 3", conn)

            records = data.to_dict('records')
            docs = [Document(page_content=record["marketing"] + record["description"]+ datetime.now().strftime('%Y-%m-%d-%H%M%S'), 
                             metadata={"id": str(record["id"]), "name": record["name"], "title":record["title"], 
                                       "image":record["image"], "video":record["video"],
                                       "gender":record["gender"], "season":record["season"], "price":record["price"],
                                       "page": "1", "chunk": "0", "source": "1-0"
                                       })
                    for record in records]
            ids = [record["id"] for record in records]

            vector_index = product_vector_index()
            vector_index.delete(ids)
            vector_index.add_documents(docs, ids=ids)

            st.toast("保存成功！", icon="✌️")

    except Exception as e:
        st.exception(e)

def select_product_page(page):
    with engine.connect() as conn:
        df = pd.read_sql(f"""
            select id as 商品编号, name as 商品名称, title as 商品标题, tags as 商品标签, image as 商品主图, video as 商品视频, gender as 商品类型, season as 适合季节, price as 商品价格, marketing as 营销文案, description as 商品描述 
            from ai_labs_product_info
            
            order by updated_at desc limit {page * 10 - 10}, 10
            """, conn)
        # df["商品主图"]=df["商品主图"].apply(lambda x: "data:image/png;base64," + image_to_base64(x))
        return df


def query_product_page_condition():
    with engine.connect() as conn:
        sql = f"""
            select id as 商品编号, name as 商品名称, title as 商品标题, tags as 商品标签, 
                   gender as 商品类型, season as 适合季节, price as 商品价格, marketing as 营销文案, description as 商品描述, 
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
                        if row.iloc[9]:
                            st.image(row.iloc[9])
                    with tabs[1]:
                        if row.iloc[10]:
                            st.video(row.iloc[10])
                with cols[1]:
                    st.write("商品名称：" + row.iloc[1])
                    st.write("商品标签：" + row.iloc[3])
                    st.write("商品类型：" + row.iloc[4])
                    st.write("适合季节：" + row.iloc[5])
                    st.write("商品价格：" + str(row.iloc[6]))
                    st.write( str(row.iloc[7]))
                with cols[2]:
                    st.write(row.iloc[8])
                cols = st.columns([0.1,0.1,0.1,0.1,0.6])
                with cols[0]:
                    # st.markdown(f"<a href='/在线试穿?id={row.iloc[0]}'>在线试穿</a>", unsafe_allow_html=True)
                    if st.button("在线试穿", key=f"try_on_{row.iloc[0]}", type="primary"):
                        st.session_state.tryon_id = row.iloc[0]
                        st.switch_page("pages/61👚在线试穿.py")
                    
                if st.session_state.rolename == "卖家":
                    with cols[1]:
                        with st.popover("修改"):
                            with st.form(f"edit_product_{row.iloc[0]}_form"):
                                columns = st.columns(3)
                                with columns[0]:
                                    product_name = st.text_input(label="商品名称", key=f"product_name_{row.iloc[0]}", value=row.iloc[1])
                                    product_gender = st.selectbox("商品类型", key=f"product_gender_{row.iloc[0]}", options=["男装","女装"], index=["男装","女装"].index(row.iloc[4]))
                                with columns[1]:
                                    product_title = st.text_input(label="商品标题", key=f"product_title_{row.iloc[0]}", value=row.iloc[2])
                                    product_season = st.selectbox("适合季节", key=f"product_season_{row.iloc[0]}", options=["春季","夏季","秋季","冬季"], index=["春季","夏季","秋季","冬季"].index(row.iloc[5]))
                                with columns[2]:
                                    product_tags = st.text_input(label="商品标签", key=f"product_tags_{row.iloc[0]}", value=row.iloc[3])
                                    product_price = st.slider("商品价格", key=f"product_price_{row.iloc[0]}", min_value=0.0, max_value=1000.0, step=0.1, value=row.iloc[6])
                                product_marketing = st.text_area(label="营销文案", key=f"product_marketing_{row.iloc[0]}", value=row.iloc[7])
                                product_description = st.text_area(label="商品描述", key=f"product_description_{row.iloc[0]}", value=row.iloc[8])
                                if st.form_submit_button("提交", type="primary", use_container_width=True):
                                    save_product_info(row.iloc[0], product_name, product_title, product_tags, "", "", "", product_gender, product_season, product_price, product_marketing, product_description)
                    with cols[2]:
                        with st.popover("评价"):
                            with st.form(f"comment_product_{row.iloc[0]}_form"):
                                rating = st.number_input("评分", key=f"product_rating_{row.iloc[0]}", min_value=1, max_value=5)
                                comment = st.text_area(label="评论", key=f"product_comment_{row.iloc[0]}")
                                if st.form_submit_button("提交", type="primary", use_container_width=True):
                                    save_product_ratings(row.iloc[0], rating, comment)

                if st.session_state.rolename == "买家":
                    with cols[1]:
                        if st.button("立即购买", type="primary", key=f"buy_product_{row.iloc[0]}"):
                            update_aigc_perm_freq(int(row.iloc[6] * 10))
                            st.toast("购买成功，感谢您的支持！", icon="💰")
                    with cols[2]:
                        if st.button("加入购物车", key=f"favorite_product_{row.iloc[0]}"):
                            st.toast("加购成功，感谢您的支持！", icon="🛒")
                    with cols[3]:
                        with st.popover("评价"):
                            with st.form(f"comment_product_{row.iloc[0]}_form"):
                                rating = st.number_input("评分", key=f"product_rating_{row.iloc[0]}", min_value=1, max_value=5)
                                comment = st.text_area(label="评论", key=f"product_comment_{row.iloc[0]}")
                                if st.form_submit_button("提交", type="primary", use_container_width=True):
                                    save_product_ratings(row.iloc[0], rating, comment)


if __name__ == '__main__':
    if st.session_state.rolename == "卖家":
        with st.expander("录入商品"):
            with st.form("add_product_info", clear_on_submit=False):
                cols = st.columns(3)
                with cols[0]:
                    product_name = st.text_input(label="商品名称")
                with cols[1]:
                    product_title = st.text_input(label="商品标题")
                with cols[2]:
                    product_tags = st.text_input(label="商品标签")
                cols = st.columns(3)
                with cols[0]:
                    product_gender = st.selectbox("商品类型",options=["男装","女装"])
                with cols[1]:
                    product_season = st.selectbox("适合季节",options=["春季","夏季","秋季","冬季"])
                with cols[2]:
                    product_price = st.slider("商品价格", min_value=0.0, max_value=1000.0, step=0.1, value=50.0)
                cols = st.columns(3)
                with cols[0]:
                    product_image = st.file_uploader("平铺图",type=["png","jpg"])
                with cols[1]:
                    product_video = st.file_uploader("视频", type=["mp4", "avi"])
                with cols[2]:
                    product_images = st.file_uploader("其他图",type=["png","jpg"], accept_multiple_files=True)
                cols = st.columns(3)
                with cols[0]:
                    marketing = st.form_submit_button("生成营销文案", use_container_width=True)
                    if marketing:
                        st.session_state.product_marketing = image_chat_answer(Image.open(product_image).convert('RGB'))
                product_marketing = st.text_area("营销文案", key="product_marketing")
                product_description = st.text_area("商品描述")

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
                    save_product_info("", product_name, product_title, product_tags, image_main_path, video_path, ",".join(images_path), product_gender, product_season, product_price, product_marketing, product_description)

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
