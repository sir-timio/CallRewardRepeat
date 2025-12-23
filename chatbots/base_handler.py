import time
import json
from abc import ABC, abstractmethod
from typing import List, Dict
import torch
from vllm import LLM as VLLModel, SamplingParams
from transformers import AutoTokenizer
from chatbots.conversation import Conversation
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


class BaseHandler(ABC):
    def __init__(
        self,
        model_name,
        template,
        # chat_template=None,
        gpu_memory_utilization=0.8,
        reduce_inference_mem=False,
        skip_llm_init: bool = False,
    ):
        self.template = template
        if os.path.exists(model_name):
            self.model_name = model_name
        else:
            if os.path.exists(f"/root/workspace/fnctod/models/{model_name}"):
                self.model_name = f"/root/workspace/fnctod/models/{model_name}"
            else:
                self.model_name = model_name
                print("trying load from hf")
        self.conversation = Conversation(template)
        if skip_llm_init:
            self.llm = None
        else:
            num_gpus = torch.cuda.device_count()
            print(f"NUM DEVICES: {num_gpus}")
            self.llm = VLLModel(
                model=self.model_name,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=10_000,
                tensor_parallel_size=num_gpus,
                enforce_eager=reduce_inference_mem,
                # max_num_seqs=1 if reduce_inference_mem else None
            )
            # if "4bit" in model_name:
            #     self.llm = VLLModel(
            #         model=self.model_name,
            #         gpu_memory_utilization=gpu_memory_utilization,
            #         quantization="bitsandbytes",
            #         load_format="bitsandbytes",
            #         dtype="bfloat16",
            #         max_model_len=10_000,
            #         )
            # else:
            #     self.llm = VLLModel(
            #             model=self.model_name,
            #             gpu_memory_utilization=gpu_memory_utilization,
            #             max_model_len=10_000,
            #         )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # if chat_template:
        #     if self.tokenizer.chat_template is not None:
        #         print("OVERRIDING CHAT TEMPLATE")
        #     self.tokenizer.chat_template = chat_template

    @abstractmethod
    def format_prompt(
        self,
        messages,
        functions=None,
        function_call=None,
        add_generation_prompt: bool = True,
    ):
        pass

    @abstractmethod
    def parse_response(
        self, response_text, required: list[str] = ["function_call", "content"]
    ):
        pass

    def generate(
        self,
        messages,
        functions=None,
        function_call=None,
        temperature: float = 0.1,
        repetition_penalty: float = 1,
        top_p: float = 1,
        max_tokens: int = 128,
        required: List[str] = ["function_call", "content"],
    ):
        formatted_prompt = self.format_prompt(
            messages, functions, function_call, add_generation_prompt=True
        )
        # print(formatted_prompt)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        outputs = self.llm.generate([formatted_prompt], sampling_params, use_tqdm=True)
        response_text = outputs[0].outputs[0].text
        parsed_response = self.parse_response(response_text, required=required)

        return parsed_response

    def batch_generate(
        self,
        batch_messages,
        batch_functions=None,
        function_call=None,
        temperature: float = 0.1,
        repetition_penalty: float = 1,
        top_p: float = 1,
        max_tokens: int = 128,
        required: List[str] = ["function_call", "content"],
        n_seqs:int = 1
    ):
        formatted_prompts = []
        for i, (messages, functions) in enumerate(zip(batch_messages, batch_functions)):
            formatted_prompt = self.format_prompt(
                messages=messages,
                functions=functions,
                function_call=function_call,
                add_generation_prompt=True,
            )
            formatted_prompts.append(formatted_prompt)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        batch_outputs = self.llm.generate(
            formatted_prompts, sampling_params, use_tqdm=False
        )
        parsed_responses = []
        for outputs in batch_outputs:
            response_text = outputs.outputs[0].text
            parsed_response = self.parse_response(response_text, required=required)
            parsed_responses.append(parsed_response)
        return parsed_responses
