#!/bin/python3

import os
import requests
import logging
import inspect

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import time
import json
import torch
from typing import Any, Dict, List, Union, Optional

from vllm import LLM as VLLModel
from vllm import SamplingParams
from chatbots.handlers.trained_model_handler import TrainedModelHandler
from chatbots.handlers.chat_template_handler import ChatTemplateHandler

import openai
import anthropic
import tiktoken

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from chatbots.utils import *
from chatbots.configs import llm_configs
from chatbots.conversation import Conversation


"""
The wrapper for ChatCompletions
"""


class chat_completion(object):
    def __init__(
        self,
        model,
        template: str,
        handler: str,
        # chat_template: str = "",
        api=False,
        system_message: str = "",
        system_template: str = "{system_message}",
        roles: List[str] = ["User", "Assistant"],
        offset: int = 20,
        colon: str = ": ",
        separators: List[str] = ["\n", "\n", "\n"],
        function_type: str = "json",
        function_call_prefix: str = "<tool_call>",
        function_call_suffix: str = "</tool_call>\n",
        gpu_memory_utilization: float = 0.8,
        verbose: bool = False,
        reduce_inference_mem: bool = False,
    ):
        self.verbose = verbose
        self.api = api
        # model_name = llm_configs[model]["model_name"]
        model_name = model
        # template = llm_configs[model]["template"]

        if handler == "trained":
            HandlerCLS = TrainedModelHandler
        elif handler == "chat_template":
            HandlerCLS = ChatTemplateHandler
        else:
            raise Exception("unknown handler")

        # HandlerCLS = llm_configs[model].get("handler", TrainedModelHandler)

        self.model_name = model_name

        if self.api:  # docker api calling
            port = llm_configs[model]["port"]
            self.url = f"http://127.0.0.1:{port}/generate"
        else:  # local inference
            self.model = HandlerCLS(model_name=model_name, template=template, gpu_memory_utilization=gpu_memory_utilization, reduce_inference_mem=reduce_inference_mem)

        self.template = template

    def complete(
        self,
        messages: List[Dict] = [],
        functions: List[Dict] = [],
        function_call: Dict = {},
        examples: List[Dict] = [],  # examples in the system instruction
        required: List[str] = ["function_call", "content"],
        temperature: float = 0.5,
        repetition_penalty: float = 1.0,
        top_p: float = 0.8,
        max_tokens: int = 64,
        n_seqs: int = 1,
        stop: List[str] = ["\n\n"],  # ["\n\n", "###", "User", "Assistant", "Example"]
    ) -> List[str]:
        if self.template == "chatgpt":
            print("under reconstruction")
            # # system messages
            # system_message = ""
            # for message in messages:
            #     if message["role"] == "system":
            #         system_message = message["content"]
            #         break

            # # construct system prompt with only examples
            # system_prompt = self.conversation.get_prompt(
            #     system_message=system_message, examples=examples
            # )

            # # replace the original system message with the one with examples
            # for message in messages:
            #     if message["role"] == "system":
            #         message["content"] = system_prompt
            #         break

            # t1 = time.time()
            # outputs = self.model.generate(
            #     prompt=messages,
            #     functions=functions,
            #     function_call=function_call,
            #     temperature=temperature,
            #     top_p=top_p,
            #     max_tokens=max_tokens,
            #     n_seqs=n_seqs,
            #     stop=stop,
            # )
            # t2 = time.time()
            # duration = t2 - t1

            # # input tokens
            # input_tokens = 0
            # for message in messages:
            #     input_tokens += len(self.model.tokenizer.encode(message["content"]))

            # # output tokens
            # for idx, output in enumerate(outputs):
            #     output["duration"] = duration
            #     output["input_tokens"] = input_tokens
            #     outputs[idx] = output
        else:  # local inference
            if self.verbose:
                print(json.dumps(messages))
            t1 = time.time()
            if self.api:
                data = {
                    "input": prompt,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "n_seqs": n_seqs,
                    "stop": stop,
                }
                response = requests.post(self.url, json=data)
                outputs = response.json()["generated_text"]
            else:
                outputs = self.model.generate(
                    messages=messages,
                    functions=functions,
                    function_call=function_call,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    n_seqs=n_seqs,
                    # stop=stop,
                )
                outputs = [outputs]  # TODO: replace this hot fix for duration and so
            t2 = time.time()
            duration = t2 - t1

            # input tokens
            input_tokens = len(
                self.model.tokenizer.apply_chat_template(
                    messages, tools=functions if len(functions) else None
                )
            )

            # cost summary
            for idx, output in enumerate(outputs):
                output["duration"] = duration
                # output["input_tokens"] = input_tokens
                outputs[idx] = output

        # chatml format: {"content": "xxx", "function": {}}
        return outputs

    def complete_batch(
        self,
        batch_messages: List[List[Dict]] = [],
        batch_functions: List[Dict] = [],
        function_call: Dict = {},
        examples: List[Dict] = [],  # examples in the system instruction
        required: List[str] = ["function_call", "content"],
        temperature: float = 0.5,
        repetition_penalty: float = 1.0,
        top_p: float = 0.8,
        max_tokens: int = 64,
        n_seqs: int = 1,
        stop: List[str] = ["\n\n"],  # ["\n\n", "###", "User", "Assistant", "Example"]
    ) -> List[str]:
        if self.template == "chatgpt":
            raise NotImplementedError
        else:  # local inference
            if self.verbose:
                print(json.dumps(batch_messages))
            t1 = time.time()
            if self.api:
                data = {
                    "input": prompt,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "n_seqs": n_seqs,
                    "stop": stop,
                }
                response = requests.post(self.url, json=data)
                outputs = response.json()["generated_text"]
            else:
                outputs = self.model.batch_generate(
                    batch_messages=batch_messages,
                    batch_functions=batch_functions,
                    function_call=function_call,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    n_seqs=n_seqs,
                    # stop=stop,
                )

            t2 = time.time()
            duration = t2 - t1

            for idx, output in enumerate(outputs):
                output["duration"] = duration
                output["input_tokens"] = self.model.tokenizer.apply_chat_template(
                    batch_messages[idx],
                    tools=batch_functions[idx] if batch_functions else None,
                )
                outputs[idx] = output

        return outputs


# class LLM:
#     def __init__(
#         self, model_name, chat_template=None, interval=0.5, timeout=10.0, exp=2, patience=10, max_interval=4,
#     ):
#         self.model_name = model_name
#         self.openai_api = True if any([x in self.model_name for x in ["gpt-3.5", "gpt-4"]]) else False
#         self.anthropic_api = True if "claude" in self.model_name else False

#         if self.openai_api:  # OPENAI API
#             self.model = OpenAI(
#                 model=model_name,
#                 interval=interval,
#                 timeout=timeout,
#                 exp=exp,
#                 patience=patience,
#                 max_interval=max_interval,
#             )
#             self.tokenizer = tiktoken.encoding_for_model(model_name)
#         elif self.anthropic_api:  # CLAUDE API
#             self.model = Claude(
#                 model=model_name,
#                 interval=interval,
#                 timeout=timeout,
#                 exp=exp,
#                 patience=patience,
#                 max_interval=max_interval,
#             )
#             self.tokenizer = None
#         else:  # HUGGINGFACE MODELS
#             self.model = VLLModel(model=model_name)
#             self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#             self.device = "cuda" if torch.cuda.is_available() else "cpu"
#             # HOTFIXED
#             tok = AutoTokenizer.from_pretrained("Team-ACE/ToolACE-2-Llama-3.1-8B")
#             self.tokenizer.chat_template = tok.chat_template
#             if not hasattr(self.tokenizer, 'chat_template') or not hasattr(self.tokenizer, 'apply_chat_template'):
#                 if chat_template is None:
#                     raise Exception("chat template is a MUST")
#                 self.tokenizer.chat_template = chat_template

#             assert 'tools' in self.tokenizer.chat_template, "'tools' in the chat_template is a MUST"


#     def generate(
#         self,
#         messages,
#         functions=None,
#         function_call=None,
#         add_generation_prompt: bool=True,
#         temperature=0.5,
#         top_p=0.5,
#         max_tokens=128,
#         n_seqs=1,
#         stop=["\n\n", "User", "Example"],
#     ):
#         # API-accessible models
#         if self.openai_api:
#             generations = self.model.generate(
#                 messages=messages,
#                 functions=functions,
#                 function_call=function_call,
#                 temperature=temperature,
#                 top_p=top_p,
#                 max_tokens=max_tokens,
#                 n=n_seqs,
#                 stop=stop,
#             )
#         elif self.anthropic_api:
#             generations = self.model.generate(
#                 prompt=messages,
#                 temperature=temperature,
#                 top_p=top_p,
#                 max_tokens=max_tokens,
#                 n=n_seqs,
#                 stop=stop,
#             )
#         else:
#             sampling_params = SamplingParams(
#                 temperature=temperature,
#                 top_p=top_p,
#                 max_tokens=max_tokens,
#                 n=n_seqs,
#                 stop=stop
#             )

#             formatted_prompt = self.tokenizer.apply_chat_template(messages, tools=functions, tokenize=True, add_generation_prompt=add_generation_prompt)
#             outputs = self.model.generate(
#                 prompt_token_ids=formatted_prompt,
#                 sampling_params=sampling_params
#             )
#             generations = [output.outputs[0].text for output in outputs]


#         return generations

#     def _get_stop_token_ids(self, stop):
#         stop = [] if stop is None else stop
#         stop = list(set(stop + ["\n", "Ċ", "ĊĊ", "<0x0A>"]))

#         try:
#             stop_token_ids = list(
#                 set(
#                     [
#                         self.tokenizer._convert_token_to_id(stop_token)
#                         for stop_token in stop
#                     ]
#                     + [self.model.config.eos_token_id]
#                 )
#             )
#         except:
#             stop_token_ids = list(
#                 set(
#                     [
#                         self.tokenizer.vocab.get(
#                             stop_token, self.tokenizer.unk_token_id
#                         )
#                         for stop_token in stop
#                     ]
#                     + [self.model.config.eos_token_id]
#                 )
#             )

#         # Check for unk_token_id and remove it from stop_token_ids list
#         if hasattr(self.tokenizer, 'unk_token_id') and self.tokenizer.unk_token_id is not None:
#             if self.tokenizer.unk_token_id in stop_token_ids:
#                 stop_token_ids.remove(self.tokenizer.unk_token_id)

#         return stop_token_ids
