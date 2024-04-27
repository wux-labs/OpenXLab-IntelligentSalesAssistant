import streamlit as st

from utils import is_cuda_available


def init_chat_config_form():
    model_option = ["ai-labs/sales-chat-1_8b", "internlm/internlm2-chat-7b"]
    if is_cuda_available():
        model_option.append("THUDM/chatglm3-6b")

    model = st.selectbox("Model", key="config_chat_model", options=model_option)
    max_tokens = st.number_input("Max Tokens", key="config_chat_max_tokens", min_value=512, max_value=4096,
                                    step=1, value=2048,
                                    help="The maximum number of tokens to generate in the chat completion.The total length of input tokens and generated tokens is limited by the model's context length.")
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature", key="config_chat_temperature", min_value=0.1, max_value=2.0,
                                value=1.0, step=0.1,
                                help="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.")
        presence_penalty = st.slider("Presence Penalty", key="config_chat_presence_penalty", min_value=-2.0,
                                        max_value=2.0, value=0.0, step=0.1,
                                        help="Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.")
    with col2:
        top_p = st.slider("Top P", key="config_chat_top_p", min_value=1.0, max_value=5.0, value=1.0, step=1.0,
                            help="An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. We generally recommend altering this or temperature but not both.")
        frequency_penalty = st.slider("Frequency Penalty", key="config_chat_frequency_penalty", min_value=-2.0,
                                        max_value=2.0, value=0.0, step=0.1,
                                        help="Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.")


def get_chat_api_base():
    return ""


def get_chat_api_key():
    return ""
