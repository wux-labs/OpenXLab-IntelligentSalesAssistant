import streamlit as st
import torch
from PIL import Image

from database.database import engine
from sqlalchemy import text

from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from common.chat import load_model_by_id, combine_history
from common.chat import config_chat_max_new_tokens, config_chat_temperature, config_chat_top_p
from common.chat import default_model
from common.product import select_product, product_vector_index

from utils import init_page_header, init_session_state, get_avatar
from utils import is_cuda_available, clear_cuda_cache, clear_streamlit_cache
from utils import global_system_prompt

from llms.sales import Sales

title = "商品咨询"
icon = "🙋🏻"
init_page_header(title, icon)
init_session_state()

product_index_directory = "products/product_index"


conversation_system_prompt = """{global_system_prompt}

有一件{product_name}

衣服的亮点包括：{product_advantage}

衣服的其他详细信息包括：
{product_info}

你需要精确获取到商品的亮点价值，激发用户的购买欲。

但请你铭记：禁止捏造数据！
"""

template = """Create a final answer to the given questions using the provided document excerpts(in no particular order) as references. ALWAYS include a "SOURCES" section in your answer including only the minimal set of sources needed to answer the question. If you are unable to answer the question, simply state that you do not know. Do not attempt to fabricate an answer and leave the SOURCES section empty.

---------

QUESTION: What  is the purpose of ARPA-H?
=========
Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more.
Source: 1-32
Content: While we’re at it, let’s make sure every American can get the health care they need. \n\nWe’ve already made historic investments in health care. \n\nWe’ve made it easier for Americans to get the care they need, when they need it. \n\nWe’ve made it easier for Americans to get the treatments they need, when they need them. \n\nWe’ve made it easier for Americans to get the medications they need, when they need them.
Source: 1-33
Content: The V.A. is pioneering new ways of linking toxic exposures to disease, already helping  veterans get the care they deserve. \n\nWe need to extend that same care to all Americans. \n\nThat’s why I’m calling on Congress to pass legislation that would establish a national registry of toxic exposures, and provide health care and financial assistance to those affected.
Source: 1-30
=========
FINAL ANSWER: The purpose of ARPA-H is to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more.
SOURCES: 1-32

---------

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""

STUFF_PROMPT = PromptTemplate(
    template=f"{global_system_prompt}\n{template}", input_variables=["summaries", "question"]
)


# @st.cache_resource
def load_product_documents(id):
    vector_index = product_vector_index(product_index_directory)
    product_documents = vector_index.search("","similarity", k=1, filter={"id":f"{id}"})
    return product_documents


@st.cache_resource
def load_chain():
    chain = load_qa_with_sources_chain(
        llm=st.session_state["ask_product_llm"],
        chain_type="stuff",
        prompt=STUFF_PROMPT,
    )
    return chain


def introduce_product(product_info):
    messages = [
        {"role": "system", "content": conversation_system_prompt.format(global_system_prompt=global_system_prompt, product_name=product_info.iloc[1], product_advantage=product_info.iloc[9], product_info=product_info.iloc[13])}
    ]
    # user_text = "你需要根据我给出的商品信息撰写一段至少500字的商品直播文案，文案内容必须基于商品信息撰写，禁止捏造内容。文案中不要提及直播间，要说本店。文案中不要给出商品的任何链接，仅介绍商品信息。你会和客户进行多轮会话，不要和客户说再见。"
    user_text = "你需要根据我给出的商品信息用500字文案详细描述一下这件服装，内容必须基于商品信息撰写，禁止捏造内容。文案中不要提及直播间，要说本店。文案中不要给出商品的任何链接，仅介绍商品信息。你会和客户进行多轮会话，不要和客户说再见。"
    answer = ""
    with st.chat_message("assistant", avatar=get_avatar("")):
        with st.spinner("处理中，请稍等..."):
            with st.empty():
                tokenizer, model, deploy = load_model_by_id(default_model)
                if deploy == "huggingface":
                    for answer, history in model.stream_chat(
                        tokenizer,
                        combine_history(messages, user_text),
                        max_new_tokens=config_chat_max_new_tokens,
                        temperature=config_chat_temperature,
                        top_p=config_chat_top_p,
                    ):
                        st.markdown(answer)
                elif deploy == "lmdeploy":
                    for item in model.stream_infer(
                        combine_history(messages, user_text),
                        max_new_tokens=config_chat_max_new_tokens,
                        temperature=config_chat_temperature,
                        top_p=config_chat_top_p,
                    ):
                        if "~" in item.text:
                            answer += item.text.replace("~", "")
                        else:
                            answer += item.text
                        st.markdown(answer)
                    st.markdown(answer)
                st.session_state["ask_product_history"].append({"role": "assistant", "content": answer})


if __name__ == '__main__':

    clear_streamlit_cache(["chat_tokenizer", "chat_model", "ask_product_history", "ask_product_llm"])

    id = None
    if "id" in st.query_params.keys():
        id = st.query_params["id"]
    elif "ask_product_id" in st.session_state.keys():
        id = st.session_state.ask_product_id
    
    if id is None:
        st.switch_page("pages/41🛍️商品管理.py")
    
    product_info = select_product(id).iloc[0]

    if "ask_product_history" not in st.session_state.keys():
        st.session_state["ask_product_history"] = [{"role": "system", "content": conversation_system_prompt.format(global_system_prompt=global_system_prompt, product_name=product_info.iloc[1], product_advantage=product_info.iloc[9], product_info=product_info.iloc[13])}]

    if "ask_product_llm" not in st.session_state.keys():
        st.session_state["ask_product_llm"] = Sales()

    with st.sidebar:
        tabs = st.tabs(["商品主图", "商品视频"])
        with tabs[0]:
            if product_info.iloc[12]:
                st.image(product_info.iloc[12])
        with tabs[1]:
            if product_info.iloc[13]:
                st.video(product_info.iloc[13])

    for message in st.session_state["ask_product_history"]:
        content = message['content']
        if message['role'] == 'user':
            with st.chat_message("user"):
                st.write(content)
        elif message['role'] == 'assistant':
            with st.chat_message("assistant", avatar=get_avatar("")):
                st.write(content)

    if len(st.session_state["ask_product_history"]) == 1:
        introduce_product(product_info)

    user_input_text = st.chat_input("您的输入...")

    if user_input_text:
        with st.chat_message("user"):
            with st.spinner("处理中，请稍等..."):
                st.write(user_input_text)

        try:
            with st.chat_message("assistant", avatar=get_avatar("")):
                with st.spinner("处理中，请稍等..."):
                    product_documents = load_product_documents(id)
                    chain = load_chain()
                    answer = chain(
                        {"input_documents": product_documents, "question": user_input_text}, return_only_outputs=True
                    )
                    st.write(answer["output_text"].split("SOURCES: ")[0])
                    st.session_state["ask_product_history"].append({"role": "user", "content": user_input_text})
                    st.session_state["ask_product_history"].append({"role": "assistant", "content": answer["output_text"].split("SOURCES: ")[0]})
        finally:
            torch.cuda.empty_cache()

