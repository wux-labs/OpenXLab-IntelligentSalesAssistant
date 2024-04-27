import streamlit as st
import os
import openai
import torch

from datetime import datetime
from database.database import engine
from sqlalchemy import text

from common.chat import init_chat_config_form, get_chat_api_base, get_chat_api_key
from utils import get_avatar, init_page_header, init_session_state
from utils import select_aigc_left_freq, update_aigc_perm_freq, use_limited
from utils import load_model_by_id

title = "智能聊天"
icon = "🤖"
init_page_header(title, icon)
init_session_state()


def select_chat_freq():
    count = 0
    try:
        with engine.connect() as conn:
            sql = text(f'''
            select count(*) from ai_labs_chat where username = :username and date_time >= current_date;
            ''')
            count = conn.execute(sql, [{'username': st.session_state.username}]).fetchone()[0]
    except Exception as e:
        st.exception(e)
    return count


def select_aigc_freq():
    st.session_state.aigc_temp_freq, st.session_state.aigc_perm_freq = select_aigc_left_freq()
    st.session_state.aigc_temp_chat = select_chat_freq()


def insert_chat(user, assistant):
    try:
        with engine.connect() as conn:
            date_time = datetime.now()
            sql = text(f'''
            insert into ai_labs_chat(model_id, username, user, assistant, date_time) values(:model_id, :username, :user, :assistant, :date_time)
            ''')
            conn.execute(sql, [{
                'model_id': st.session_state.config_chat_model,
                'username': st.session_state.username,
                'user': user,
                'assistant': assistant,
                'date_time': date_time
            }])
            conn.commit()
    except Exception as e:
        st.exception(e)


def select_chat():
    with engine.connect() as conn:
        sql = text("""
            select id,model_id,user,assistant from (select * from ai_labs_chat where username = :username order by id desc limit :showlimit) temp order by id
        """)
        return conn.execute(sql, [{
            'username': st.session_state.username,
            'showlimit': st.session_state.showlimit
        }]).fetchall()


def select_chat_lastn():
    with engine.connect() as conn:
        sql = text("""
            select id,model_id,user,assistant from (select * from ai_labs_chat where model_id = :model_id and username = :username order by id desc limit :history) temp order by id
        """)
        return conn.execute(sql, [{
            'model_id': st.session_state.config_chat_model,
            'username': st.session_state.username,
            'history': st.session_state.config_chat_history_messsages
        }]).fetchall()


def cache_chat(user_input_text):
    with st.chat_message("user"):
        st.write(user_input_text)
    
    with st.chat_message("assistant", avatar=get_avatar(st.session_state.config_chat_model)):
        with st.spinner("处理中，请稍等..."):
            answer = ""
            messages = []
            for id, model_id, user, assistant in select_chat_lastn():
                messages.append({
                    "role": "user",
                    "content": user
                })
                messages.append({
                    "role": "assistant",
                    "content": assistant
                })
            messages.append({
                "role": "user",
                "content": user_input_text
            })

            with st.empty():
                if st.session_state.config_chat_model == "ai-labs/sales-chat-1_8b":
                    tokenizer, model = load_model_by_id(st.session_state.config_chat_model)
                    if st.session_state.config_chat_stream:
                        for answer, history in model.stream_chat(
                            tokenizer,
                            user_input_text,
                            max_length=st.session_state.config_chat_max_tokens,   # 最多生成字数
                            temperature=st.session_state.config_chat_temperature, # 温度
                            top_p=st.session_state.config_chat_top_p,             # 采样概率
                        ):
                            st.write(answer)
                    else:
                        answer, history = model.chat(
                            tokenizer,
                            user_input_text
                        )
                        st.write(answer)
                elif st.session_state.config_chat_model == "internlm/internlm2-chat-7b":
                    tokenizer, model = load_model_by_id(st.session_state.config_chat_model)
                    if st.session_state.config_chat_stream:
                        for answer, history in model.stream_chat(
                            tokenizer,
                            user_input_text,
                            max_length=st.session_state.config_chat_max_tokens,   # 最多生成字数
                            temperature=st.session_state.config_chat_temperature, # 温度
                            top_p=st.session_state.config_chat_top_p,             # 采样概率
                        ):
                            st.write(answer)
                    else:
                        answer, history = model.chat(
                            tokenizer,
                            user_input_text
                        )
                        st.write(answer)
                elif st.session_state.config_chat_model == "THUDM/chatglm3-6b":
                    tokenizer, model = load_model_by_id(st.session_state.config_chat_model)
                    if st.session_state.config_chat_stream:
                        for answer, history in model.stream_chat(
                            tokenizer,
                            user_input_text,
                            max_tokens=st.session_state.config_chat_max_tokens,   # 最多生成字数
                            temperature=st.session_state.config_chat_temperature, # 温度
                            top_p=st.session_state.config_chat_top_p,             # 采样概率
                            frequency_penalty=st.session_state.config_chat_frequency_penalty,
                            presence_penalty=st.session_state.config_chat_presence_penalty,
                        ):
                            st.write(answer)
                    else:
                        answer, history = model.chat(
                            tokenizer,
                            user_input_text
                        )
                        st.write(answer)
                else:
                    openai.api_base = get_chat_api_base()
                    openai.api_key = get_chat_api_key()
                    if st.session_state.config_chat_stream:
                        for chunk in openai.ChatCompletion.create(
                            model=st.session_state.config_chat_model,
                            messages=messages,
                            max_tokens=st.session_state.config_chat_max_tokens,
                            temperature=st.session_state.config_chat_temperature,
                            top_p=st.session_state.config_chat_top_p,
                            presence_penalty=st.session_state.config_chat_presence_penalty,
                            frequency_penalty=st.session_state.config_chat_frequency_penalty,
                            stream=st.session_state.config_chat_stream
                        ):
                            if hasattr(chunk.choices[0].delta, "content"):
                                answer = answer + chunk.choices[0].delta.content
                                st.write(answer)
                    else:
                        response = openai.ChatCompletion.create(
                            model=st.session_state.config_chat_model,
                            messages=messages,
                            max_tokens=st.session_state.config_chat_max_tokens,
                            temperature=st.session_state.config_chat_temperature,
                            top_p=st.session_state.config_chat_top_p,
                            presence_penalty=st.session_state.config_chat_presence_penalty,
                            frequency_penalty=st.session_state.config_chat_frequency_penalty,
                            stream=st.session_state.config_chat_stream
                        )
                        if hasattr(response.choices[0].message, "content"):
                            answer = answer + response.choices[0].message.content
                            st.write(answer)
            if answer != "":
                insert_chat(user_input_text, answer)


if __name__ == '__main__':

    with st.sidebar:
        tabs = st.tabs(["模型设置"])
        with tabs[0]:
            init_chat_config_form()
            st.slider("History Messages", key="config_chat_history_messsages", min_value=0, max_value=10, value=0,
                step=1)
            st.toggle("Streaming", key="config_chat_stream", value=True)

    for id, model_id, user, assistant in select_chat():
        with st.chat_message("user"):
            st.write(user)
            
        with st.chat_message("assistant", avatar=get_avatar(model_id)):
            st.write(assistant)

    user_input_text = st.chat_input("您的输入...")

    if user_input_text:
        cache_chat(user_input_text)
        torch.cuda.empty_cache()
