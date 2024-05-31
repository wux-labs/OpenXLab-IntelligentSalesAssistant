import streamlit as st
import os
from datetime import datetime
from database.database import engine
from sqlalchemy import text

from common.draw import init_draw_config_form, save_draw_image
from utils import init_page_header, init_session_state, get_avatar
from utils import select_aigc_left_freq, update_aigc_perm_freq, use_limited, check_use_limit
from utils import is_cuda_available, clear_cuda_cache, clear_streamlit_cache
 
title = "图片生成"
icon = "🎨"
init_page_header(title, icon)
init_session_state()


def select_images_freq():
    count = 0
    try:
        with engine.connect() as conn:
            sql = text(f'''
            select count(*) from ai_labs_images where username = :username and date_time >= current_date;
            ''')
            count = conn.execute(sql, [{'username': st.session_state.username}]).fetchone()[0]
    except Exception as e:
        st.exception(e)
    return count


def select_aigc_freq():
    st.session_state.aigc_temp_freq, st.session_state.aigc_perm_freq = select_aigc_left_freq()
    st.session_state.aigc_temp_images = select_images_freq()


def insert_images(user, assistant):
    try:
        with engine.connect() as conn:
            date_time = datetime.now()
            sql = text(f'''
            insert into ai_labs_images(username, user, assistant, date_time) values(:username, :user, :assistant, :date_time)
            ''')
            conn.execute(sql, [{
                'username': st.session_state.username,
                'user': user,
                'assistant': assistant,
                'date_time': date_time
            }])
            conn.commit()
    except Exception as e:
        st.error(e)


def select_images():
    with engine.connect() as conn:
        sql = text("""
            select id,user,assistant from (select * from ai_labs_images where username = :username order by id desc limit :showlimit) temp order by id
        """)
        return conn.execute(sql, [{
            'username': st.session_state.username,
            'showlimit': st.session_state.showlimit
        }]).fetchall()

def cache_draw(user_input_text):
    try:
        localfile = save_draw_image(user_input_text)
        st.image(localfile)
        insert_images(user_input_text, localfile)
    except Exception as ex:
        st.exception(ex)

if __name__ == '__main__':
    
    clear_streamlit_cache(["chat_tokenizer", "chat_model", "stable_diffusion_model"])

    select_aigc_freq()

    with st.sidebar:
        tabs = st.tabs(["图片设置"])
        with tabs[0]:
            init_draw_config_form()
        if check_use_limit:
            st.info(f"免费次数已用：{min(st.session_state.aigc_temp_freq, st.session_state.aigc_temp_images)}/{st.session_state.aigc_temp_freq} 次。\n\r永久次数剩余：{st.session_state.aigc_perm_freq} 次。", icon="🙂")

    for id, user, assistant in select_images():
        with st.chat_message("user"):
            st.write(user)
        with st.chat_message("assistant", avatar=get_avatar(st.session_state.config_image_model)):
            st.image(assistant)

    user_input_text = st.chat_input("您的输入...")

    if user_input_text:
        with st.chat_message("user"):
            st.write(user_input_text)
        
        with st.chat_message("assistant", avatar=get_avatar(st.session_state.config_image_model)):
            if check_use_limit and st.session_state.aigc_temp_images >= st.session_state.aigc_temp_freq and st.session_state.aigc_perm_freq < 1:
                use_limited()
            else:
                with st.spinner("处理中，请稍等..."):
                    # st.image("statics/images/loading.gif")
                    cache_draw(user_input_text)
                    if check_use_limit and st.session_state.aigc_temp_images >= st.session_state.aigc_temp_freq:
                        update_aigc_perm_freq(-1)
                    select_aigc_freq()
                clear_cuda_cache()
