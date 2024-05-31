import streamlit as st
import os
import openai

from datetime import datetime
from database.database import engine
from sqlalchemy import text

from common.chat import init_chat_config_form, get_chat_api_base, get_chat_api_key, internlm2_models, load_model_by_id, combine_history
from utils import init_page_header, init_session_state, get_avatar
from utils import select_aigc_left_freq, update_aigc_perm_freq, use_limited, check_use_limit
from utils import is_cuda_available, clear_cuda_cache, clear_streamlit_cache
from utils import global_system_prompt

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
        if check_use_limit and st.session_state.aigc_temp_chat >= st.session_state.aigc_temp_freq and st.session_state.aigc_perm_freq < 1:
            use_limited()
        else:
            with st.spinner("处理中，请稍等..."):
                answer = ""
                messages = [{
                        "role": "system",
                        "content": global_system_prompt
                    }]
                if st.session_state.config_chat_history_messsages > 0:
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
                    if st.session_state.config_chat_model in internlm2_models:
                        tokenizer, model, deploy = load_model_by_id(st.session_state.config_chat_model)
                        if deploy == "huggingface":
                            if st.session_state.config_chat_stream:
                                for answer, history in model.stream_chat(
                                    tokenizer,
                                    combine_history(messages, user_input_text),
                                    max_new_tokens=st.session_state.config_chat_max_tokens,   # 最多生成字数
                                    temperature=st.session_state.config_chat_temperature, # 温度
                                    top_p=st.session_state.config_chat_top_p,             # 采样概率
                                ):
                                    st.markdown(answer)
                            else:
                                answer, history = model.chat(
                                    tokenizer,
                                    combine_history(messages, user_input_text),
                                    max_new_tokens=st.session_state.config_chat_max_tokens,   # 最多生成字数
                                    temperature=st.session_state.config_chat_temperature, # 温度
                                    top_p=st.session_state.config_chat_top_p,             # 采样概率
                                )
                                st.markdown(answer)
                        elif deploy == "lmdeploy":
                            if st.session_state.config_chat_stream:
                                for item in model.stream_infer(
                                    combine_history(messages, user_input_text),
                                    max_new_tokens=st.session_state.config_chat_max_tokens,   # 最多生成字数
                                    temperature=st.session_state.config_chat_temperature, # 温度
                                    top_p=st.session_state.config_chat_top_p,             # 采样概率
                                ):
                                    if "~" in item.text:
                                        answer += item.text.replace("~", "")
                                    else:
                                        answer += item.text
                                    st.markdown(answer)
                                st.markdown(answer)
                            else:
                                answer = model.chat(
                                    combine_history(messages, user_input_text),
                                    max_new_tokens=st.session_state.config_chat_max_tokens,   # 最多生成字数
                                    temperature=st.session_state.config_chat_temperature, # 温度
                                    top_p=st.session_state.config_chat_top_p,             # 采样概率
                                ).response.text
                                st.markdown(answer)
                    elif st.session_state.config_chat_model == "THUDM/chatglm3-6b":
                        tokenizer, model, deploy = load_model_by_id(st.session_state.config_chat_model)
                        if st.session_state.config_chat_stream:
                            for answer, history in model.stream_chat(
                                tokenizer,
                                user_input_text,
                                history=messages,
                                max_tokens=st.session_state.config_chat_max_tokens,   # 最多生成字数
                                temperature=st.session_state.config_chat_temperature, # 温度
                                top_p=st.session_state.config_chat_top_p,             # 采样概率
                                frequency_penalty=st.session_state.config_chat_frequency_penalty,
                                presence_penalty=st.session_state.config_chat_presence_penalty,
                            ):
                                st.markdown(answer)
                        else:
                            answer, history = model.chat(
                                tokenizer,
                                user_input_text,
                                history=messages,
                                max_tokens=st.session_state.config_chat_max_tokens,   # 最多生成字数
                                temperature=st.session_state.config_chat_temperature, # 温度
                                top_p=st.session_state.config_chat_top_p,             # 采样概率
                                frequency_penalty=st.session_state.config_chat_frequency_penalty,
                                presence_penalty=st.session_state.config_chat_presence_penalty,
                            )
                            st.markdown(answer)
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
                                    st.markdown(answer)
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
                                st.markdown(answer)
                if answer != "":
                    insert_chat(user_input_text, answer)
                    if check_use_limit and st.session_state.aigc_temp_chat >= st.session_state.aigc_temp_freq:
                        update_aigc_perm_freq(-1)
                    select_aigc_freq()
            clear_cuda_cache()


if __name__ == '__main__':

    clear_streamlit_cache(["chat_tokenizer", "chat_model"])

    select_aigc_freq()

    with st.sidebar:
        tabs = st.tabs(["模型设置"])
        with tabs[0]:
            init_chat_config_form()
            st.slider("History Messages", key="config_chat_history_messsages", min_value=0, max_value=10, value=3,
                step=1)
            st.toggle("Streaming", key="config_chat_stream", value=True)
        if check_use_limit:
            st.info(f"免费次数已用：{min(st.session_state.aigc_temp_freq, st.session_state.aigc_temp_chat)}/{st.session_state.aigc_temp_freq} 次。\n\r永久次数剩余：{st.session_state.aigc_perm_freq} 次。", icon="🙂")

    for id, model_id, user, assistant in select_chat():
        with st.chat_message("user"):
            st.write(user)
            
        with st.chat_message("assistant", avatar=get_avatar(model_id)):
            st.write(assistant)

    user_input_text = st.chat_input("您的输入...")

    if user_input_text:
        cache_chat(user_input_text)
