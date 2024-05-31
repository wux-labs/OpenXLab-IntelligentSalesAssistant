import json
import streamlit as st
from typing import Any, AsyncIterator, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.messages import (
    BaseMessage,
)
from langchain_core.pydantic_v1 import Field
from langchain_core.outputs import GenerationChunk

import os
import yaml

from common.chat import load_model_by_id, combine_history
from common.chat import config_chat_max_new_tokens, config_chat_temperature, config_chat_top_p
from common.chat import default_model


def tool_config_from_file(tool_name, directory="tools/"):
    for filename in os.listdir(directory):
        if filename.endswith('.yaml') and tool_name in filename:
            file_path = os.path.join(directory, filename)
            with open(file_path, encoding='utf-8') as f:
                return yaml.safe_load(f)
    return None


class Sales(LLM):
    model_kwargs: Optional[dict] = None
    prefix_messages: List[BaseMessage] = []

    history: List = []
    tool_names: List = []
    has_search: bool = False
    use_tool: str = None

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "Sales"

    @property
    def _invocation_params(self) -> dict:
        params = {
            "do_sample": True,
            "max_new_tokens": config_chat_max_new_tokens,
            "temperature": config_chat_temperature,
            "top_p": config_chat_top_p,
        }
        return {**params, **(self.model_kwargs or {})}

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        tokenizer, model, deploy = load_model_by_id(default_model)
        if deploy == "huggingface":
            response, history = model.chat(
                tokenizer,
                combine_history(self.history, prompt),
                **self._invocation_params
            )
        elif deploy == "lmdeploy":
            response = model.chat(
                combine_history(self.history, prompt),
                **self._invocation_params
            ).response.text
        return response
    

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        tokenizer, model, deploy = load_model_by_id(default_model)
        if deploy == "huggingface":
            async with model.chat(
                tokenizer,
                combine_history(self.history, prompt),
                **self._invocation_params
            ) as response, self.history:
                return response
        elif deploy == "lmdeploy":
            async with model.chat(
                combine_history(self.history, prompt),
                **self._invocation_params
            ).response.text as response:
                return response


    def _handle_sse_line(line: str) -> Optional[GenerationChunk]:
        try:
            return GenerationChunk(
                text=line,
            )
        except Exception:
            return None

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        tokenizer, model, deploy = load_model_by_id(default_model)
        if deploy == "huggingface":
            current_length = 0
            for response, history in model.stream_chat(
                tokenizer,
                combine_history(self.history, prompt),
                **self._invocation_params
            ):
                content = response[current_length:]
                current_length = len(response)
                chunk = self._handle_sse_line(content)
                yield chunk
                if run_manager:
                    run_manager.on_llm_new_token(chunk)
        elif deploy == "lmdeploy":
            for item in model.stream_infer(
                combine_history(self.history, prompt),
                **self._invocation_params
            ):
                chunk = self._handle_sse_line(item.text)
                yield chunk
                if run_manager:
                    run_manager.on_llm_new_token(chunk)


    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        tokenizer, model, deploy = load_model_by_id(default_model)
        if deploy == "huggingface":
            current_length = 0
            history = []
            async with model.stream_chat(
                tokenizer,
                combine_history(self.history, prompt),
                **self._invocation_params
            ) as response, history:
                content = response[current_length:]
                current_length = len(response)
                chunk = self._handle_sse_line(content)
                yield chunk
                if run_manager:
                    await run_manager.on_llm_new_token(chunk)
        elif deploy == "lmdeploy":
            async with model.stream_infer(
                combine_history(self.history, prompt),
                **self._invocation_params
            ) as item:
                chunk = self._handle_sse_line(item.text)
                yield chunk
                if run_manager:
                    await run_manager.on_llm_new_token(chunk)
