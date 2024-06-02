import streamlit as st
import openai
import os

import torch
from transformers import AutoTokenizer

from utils import is_cuda_available, is_cuda_enough
from utils import cuda_size_24gb, cuda_size_40gb

internlm2_models = ["internlm/internlm2-chat-20b"] if is_cuda_enough(cuda_size_40gb) else ["internlm/internlm2-chat-7b"]

# "ai-labs/sales-chat-7b"
default_model = internlm2_models[0]

deepinfra_models = ["meta-llama/Llama-2-70b-chat-hf", "meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-7b-chat-hf",
                    "codellama/CodeLlama-34b-Instruct-hf", "jondurbin/airoboros-l2-70b-gpt4-1.4.1",
                    "mistralai/Mistral-7B-Instruct-v0.1", "mistralai/Mixtral-8x7B-Instruct-v0.1"]

config_chat_max_new_tokens = 2048
config_chat_temperature = 0.1
config_chat_top_p = 0.7

def init_chat_config_form():
    model_option = internlm2_models + deepinfra_models

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
    return os.environ.get("CHAT_OPENAI_BASE")


def get_chat_api_key():
    return ""


def load_model_by_id(model_id_or_path, **kwargs):
    if "chat_model" not in st.session_state.keys():
        st.session_state["chat_tokenizer"] = AutoTokenizer.from_pretrained("models/" + model_id_or_path,
                                                                           trust_remote_code=True)
        if is_cuda_available():
            if is_cuda_enough(cuda_size_24gb): # 40950
                from transformers import AutoModel
                st.session_state["chat_model"] = AutoModel.from_pretrained("models/" + model_id_or_path,
                                                                        trust_remote_code=True).half().eval().cuda()
                st.session_state["chat_deploy"] = "huggingface"
            else:
                from lmdeploy import TurbomindEngineConfig, pipeline
                model_format = "hf"
                if model_id_or_path.endswith("-4bit"):
                    model_format = "awq"
                backend_config = TurbomindEngineConfig(model_format=model_format, session_len=32768, cache_max_entry_count=0.4)
                st.session_state["chat_model"] = pipeline("models/" + model_id_or_path, backend_config=backend_config, model_name="internlm2")
                st.session_state["chat_tokenizer"] = None
                st.session_state["chat_deploy"] = "lmdeploy"

        else:
            from bigdl.llm.transformers import AutoModel
            st.session_state["chat_model"] = AutoModel.from_pretrained("models/" + model_id_or_path,
                                                                       load_in_4bit=True,
                                                                       trust_remote_code=True).eval()
            st.session_state["chat_deploy"] = "bigdl"
    return st.session_state["chat_tokenizer"], st.session_state["chat_model"], st.session_state["chat_deploy"]


system_prompt = "<s><|im_start|>system\n{system}<|im_end|>\n"
user_prompt = "<|im_start|>user\n{user}<|im_end|>\n"
assistant_prompt = "<|im_start|>assistant\n{assistant}<|im_end|>\n"
cur_query_prompt = "<|im_start|>user\n{user}<|im_end|>\n\
    <|im_start|>assistant\n"


def combine_history(messages, prompt):
    total_prompt = ""
    for message in messages:
        cur_content = message['content']
        if message['role'] == 'system':
            cur_prompt = system_prompt.format(system=cur_content)
        elif message['role'] == 'user':
            cur_prompt = user_prompt.format(user=cur_content)
        elif message['role'] == 'assistant':
            cur_prompt = assistant_prompt.format(assistant=cur_content)
        else:
            raise RuntimeError
        total_prompt += cur_prompt
    total_prompt = total_prompt + cur_query_prompt.format(user=prompt)
    return total_prompt


def translate_to_english(prompt):
    answer = ""
    try:
        openai.api_base = get_chat_api_base()
        openai.api_key = get_chat_api_key()
        chunk = openai.ChatCompletion.create(
            model="jondurbin/airoboros-l2-70b-gpt4-1.4.1",
            messages=[
                {"role": "system", "content": "请将以下内容翻译成英文，如果已经是英文了就直接返回，内容如下："},
                {"role": "user", "content": prompt}
            ]
        )
        if hasattr(chunk.choices[0].message, "content"):
            answer = chunk.choices[0].message.content
        return answer
    except:
        tokenizer, model, deploy = load_model_by_id(default_model)
        if deploy == "huggingface":
            answer, history = model.chat(
                tokenizer,
                prompt,
                history=[("请将以下内容翻译成英文，如果已经是英文了就直接返回，内容如下：", '')]
            )
        elif deploy == "lmdeploy":
            answer = model.chat(
                combine_history([{"role": "system", "content": "请将以下内容翻译成英文，如果已经是英文了就直接返回，内容如下："}], prompt),
            ).response.text
        return answer
