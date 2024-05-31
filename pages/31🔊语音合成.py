import streamlit as st
import io
import os

from audio_recorder_streamlit import audio_recorder
from pydub import AudioSegment

from datetime import datetime
from database.database import engine
from sqlalchemy import text

import openai

from common.chat import init_chat_config_form, get_chat_api_base, get_chat_api_key, internlm2_models, load_model_by_id, combine_history
from common.voice import init_voice_config_form, voice_to_text_remote, voice_to_text_local, text_to_voice
from utils import init_page_header, init_session_state, get_avatar
from utils import select_aigc_left_freq, update_aigc_perm_freq, use_limited, check_use_limit
from utils import is_cuda_available, clear_cuda_cache, clear_streamlit_cache
from utils import global_system_prompt


title = "语音合成"
icon = "🔊"
init_page_header(title, icon)
init_session_state()

localdir = f"users/{st.session_state.username}/records"


def select_voice_freq():
    count = 0
    try:
        with engine.connect() as conn:
            sql = text(f'''
            select count(*) from ai_labs_voice where username = :username and date_time >= current_date;
            ''')
            count = conn.execute(sql, [{'username': st.session_state.username}]).fetchone()[0]
    except Exception as e:
        st.exception(e)
    return count


def select_aigc_freq():
    st.session_state.aigc_temp_freq, st.session_state.aigc_perm_freq = select_aigc_left_freq()
    st.session_state.aigc_temp_voice = select_voice_freq()


def insert_voice(user_voice, user_text, assistant_voice, assistant_text):
    try:
        with engine.connect() as conn:
            date_time = datetime.now()
            sql = text(f'''
            insert into ai_labs_voice(username, user_voice, user_text, assistant_voice, assistant_text, date_time) 
            values(:username, :user_voice, :user_text,:assistant_voice,  :assistant_text, :date_time)
            ''')
            conn.execute(sql, [{
                'username': st.session_state.username,
                'user_voice': user_voice,
                'user_text': user_text,
                'assistant_voice': assistant_voice,
                'assistant_text': assistant_text,
                'date_time': date_time
            }])
            conn.commit()
    except Exception as e:
        st.exception(e)


def select_voice():
    with engine.connect() as conn:
        sql = text("""
            select user_voice, user_text, assistant_voice, assistant_text from (select * from ai_labs_voice where username = :username order by id desc limit :showlimit) temp order by id
        """)
        return conn.execute(sql, [{
            'username': st.session_state.username,
            'showlimit': st.session_state.showlimit
        }]).fetchall()


def cache_voice(user_voice_file, user_input_text):
    user_input = user_input_text
    with st.chat_message("user"):
        with st.spinner("处理中，请稍等..."):
            if user_voice_file is not None:
                st.audio(user_voice_file, format="wav")
                if st.session_state.config_voice_model_type == "远程":
                    user_input = voice_to_text_remote(localdir, filename)
                else:
                    user_input = voice_to_text_local(localdir, filename)
                if st.session_state.config_voice_show_text:
                    st.write(user_input)
            else:
                st.write(user_input)

    with st.chat_message("assistant", avatar=get_avatar("myshell/melotts")):
        if check_use_limit and st.session_state.aigc_temp_voice >= st.session_state.aigc_temp_freq and st.session_state.aigc_perm_freq < 1:
            use_limited()
        else:
            with st.spinner("处理中，请稍等..."):
                if st.session_state.config_voice_voice_type == "对话":
                    messages = [{
                            "role": "system",
                            "content": global_system_prompt
                        }]
                    if st.session_state.config_chat_model in internlm2_models:
                        tokenizer, model, deploy = load_model_by_id(st.session_state.config_chat_model)
                        if deploy == "huggingface":
                            assistant_text, history = model.chat(
                                tokenizer,
                                combine_history(messages, user_input),
                            )
                        elif deploy == "lmdeploy":
                            assistant_text = model.chat(
                                combine_history(messages, user_input),
                            ).response.text
                    elif st.session_state.config_chat_model == "THUDM/chatglm3-6b":
                        tokenizer, model, deploy = load_model_by_id(st.session_state.config_chat_model)
                        assistant_text, history = model.chat(
                            tokenizer,
                            user_input,
                            messages
                        )
                    else:
                        messages.append({
                            "role": "user",
                            "content": user_input
                        })
                        openai.api_base = get_chat_api_base()
                        openai.api_key = get_chat_api_key()
                        response = openai.ChatCompletion.create(
                            model=st.session_state.config_chat_model,
                            messages=messages,
                            max_tokens=st.session_state.config_chat_max_tokens,
                            temperature=st.session_state.config_chat_temperature,
                            top_p=st.session_state.config_chat_top_p,
                            presence_penalty=st.session_state.config_chat_presence_penalty,
                            frequency_penalty=st.session_state.config_chat_frequency_penalty
                        )
                        if hasattr(response.choices[0].message, "content"):
                            assistant_text = response.choices[0].message.content
                else:
                    assistant_text = user_input

                assistant_voice = text_to_voice(assistant_text)
                st.audio(assistant_voice, format="audio/mp3")
                if st.session_state.config_voice_show_text:
                    st.write(assistant_text)
                insert_voice(user_voice_file, user_input, assistant_voice, assistant_text)
                if check_use_limit and st.session_state.aigc_temp_voice >= st.session_state.aigc_temp_freq:
                    update_aigc_perm_freq(-1)
                select_aigc_freq()
            clear_cuda_cache()


if __name__ == '__main__':

    clear_streamlit_cache(["chat_tokenizer", "chat_model", "whisper_model_base", "whisper_model_small", "whisper_model_medium", "whisper_model_large"])

    select_aigc_freq()
    
    with st.sidebar:
        tabs = st.tabs(["模型设置", "语音设置"])
        with tabs[0]:
            init_chat_config_form()
        with tabs[1]:
            init_voice_config_form()
            st.selectbox("合成类型", key="config_voice_voice_type", options=["复述", "对话"])
            st.toggle("显示文字", key="config_voice_show_text")

        cols = st.columns(5)
        with cols[2]:
            audio_bytes = audio_recorder(text="", pause_threshold=2.5, icon_size='2x', sample_rate=16000)
        if check_use_limit:
            st.info(f"免费次数已用：{min(st.session_state.aigc_temp_freq, st.session_state.aigc_temp_voice)}/{st.session_state.aigc_temp_freq} 次。\n\r永久次数剩余：{st.session_state.aigc_perm_freq} 次。", icon="🙂")

    for user_voice, user_text, assistant_voice, assistant_text in select_voice():
        try:
            with st.chat_message("user"):
                if user_voice:
                    st.audio(user_voice, format="audio/mp3")
                    if st.session_state.config_voice_show_text:
                        st.write(user_text)
                else:
                    st.write(user_text)
            with st.chat_message("assistant", avatar=get_avatar("myshell/melotts")):
                st.audio(assistant_voice, format="audio/mp3")
                if st.session_state.config_voice_show_text:
                    st.write(assistant_text)
        except:
            pass

    user_input_text = st.chat_input("您的输入...")

    if user_input_text:
        cache_voice(None, user_input_text)
    elif audio_bytes:
        os.makedirs(localdir, exist_ok=True)
        filename = f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}.wav"
        filepath = f"{localdir}/{filename}"

        audio_segment = AudioSegment.from_wav(io.BytesIO(audio_bytes))
        audio_segment.export(filepath, format='wav')

        cache_voice(filepath, None)
