import dataclasses
from enum import auto, IntEnum
from typing import List, Any, Dict, Union
import json
import re

import openai
import anthropic


class Conversation(object):
    def __init__(
        self,
        template_name: str = "",
        system_message: str = "",
        system_template: str = "{system_message}",
        roles: List[str] = ["User", "Assistant"],
        offset: int = 20,
        colon: str = ": ",
        separators: List[str] = ["\n", "\n", "\n"],
        function_type: str = "json",
        function_call_prefix: str = "<function_call>",
        function_call_suffix: str = "</function_call>",
        verbose: bool = False,
    ):
        self.template_name = template_name
        self._verbose = verbose
        self.offset = offset

        # function call template
        self.function_type = function_type
        self.function_call_prefix = function_call_prefix
        self.function_call_suffix = function_call_suffix

        # conversation template
        # assert self.template_name is not None
        if self.template_name == "chatgpt":
            self.system_template = "{system_message}"
            self.separators = ("\n\n", "\n", "\n")
            self.roles = ("user", "assistant")
            self.colon = ": "
        elif self.template_name == "claude":
            self.system_template = "{system_message}"
            self.separators = ("\n\n", "\n\n", "\n\n")
            self.roles = (anthropic.HUMAN_PROMPT, anthropic.AI_PROMPT)
            self.colon = ""
        elif self.template_name == "qwen":
            self.system_template = "<|im_start|>system\n{system_message}<|im_end|>\n"
            self.separators = ("\n", "\n", "\n")
            self.roles = ("<|im_start|>user", "<|im_start|>assistant")
            self.colon = "\n"
            self.end_token = "<|im_end|>"
            self.function_call_prefix = "<tool_call>\n"
            self.function_call_suffix = "\n</tool_call>"
        elif self.template_name == "qwen3":
            self.system_template = "<|im_start|>system\n{system_message}<|im_end|>\n"
            self.separators = ("\n", "\n", "\n")
            self.roles = (
                "<|im_start|>user",
                "<|im_start|>assistant<think>\n\n</think>\n",
            )
            self.colon = "\n"
            self.end_token = "<|im_end|>"
            self.function_call_prefix = "<tool_call>\n"
            self.function_call_suffix = "\n</tool_call>"
        elif self.template_name == "qwen3-think":
            self.system_template = "<|im_start|>system\n{system_message}<|im_end|>\n"
            self.separators = ("\n", "\n", "\n")
            self.roles = ("<|im_start|>user", "<|im_start|>assistant")
            self.colon = "\n"
            self.end_token = "<|im_end|>"
            self.function_call_prefix = "<tool_call>\n"
            self.function_call_suffix = "\n</tool_call>"
        elif self.template_name == "xlam":
            self.system_template = "<|im_start|>system\n{system_message}<|im_end|>\n"
            self.separators = ("\n", "\n", "\n")
            self.roles = ("<|im_start|>user", "<|im_start|>assistant")
            self.colon = "\n"
            self.end_token = "<|im_end|>"
            self.function_call_prefix = "["
            self.function_call_suffix = "]"
        elif (
            self.template_name == "llama31"
            or self.template_name == "llama3.1"
            or self.template_name == "llama"
        ):
            self.system_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|im_end|>\n"
            self.separators = ("\n", "\n", "\n")
            self.roles = (
                "<|start_header_id|>user<|end_header_id|>",
                "<|start_header_id|>assistant<|end_header_id|>",
            )
            self.colon = "\n"
            self.end_token = "<|eot_id|>"
            self.function_call_prefix = "\n<|python_tag|>"
            self.function_call_suffix = "\n</tool_call>"
        elif self.template_name == "toolace":
            self.system_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|im_end|>\n"
            self.separators = ("\n", "\n", "\n")
            self.roles = (
                "<|start_header_id|>user<|end_header_id|>",
                "<|start_header_id|>assistant<|end_header_id|>",
            )
            self.colon = "\n"
            self.end_token = "<|eot_id|>"
            self.function_call_prefix = "["
            self.function_call_suffix = "]"
        else:  # customized
            self.system_message = system_message
            self.system_template = system_template
            self.roles = roles
            self.colon = colon
            self.separators = separators

    def get_prompt(
        self,
        system_message=None,
        messages: List[Dict] = [],
        functions: List[Dict] = [],
        function_call: Dict = {},
        examples: List[List] = [],
        add_generation_prompt: bool = True,
        add_fc_generation_prompt: bool = False,
    ) -> Union[List, str]:
        system_messages = []

        # part 1: system instruction
        if system_message:
            system_messages.append(system_message)
        else:
            system_messages.append(self.system_message)

        # part 2: functions
        if functions:
            function_prompt = self.get_functions(functions=functions)
            system_messages.append(function_prompt)

        # part 3: examples
        if examples:
            example_prompt = self.get_examples(examples=examples)
            system_messages.append(example_prompt)

        # combine them, fill in the template
        system_prompt = "\n\n".join(system_messages)
        system_prompt = self.system_template.format(system_message=system_prompt)

        # part 4: the current conversation, consisting of the current turn, with the function_call prefix
        conversation = self.get_conversation(
            messages=messages,
            function_call=function_call,
            add_generation_prompt=add_generation_prompt,
            add_fc_generation_prompt=add_fc_generation_prompt,
        )
        ret = system_prompt + self.separators[0] + conversation
        return ret

    def get_examples(self, examples):
        example_prompts = []
        for example in examples:
            if self.template_name != "llama2":
                example_prompt = self.get_conversation(example)
            else:  # not use instruction and system tokens in the examples, only for llama2
                example_prompt = self.get_conversation(
                    messages=example,
                    function_call={},
                    roles=["User", "Assistant"],
                    colon=": ",
                    separators=["", "\n", "\n"],
                )
            example_prompt = "<EXAMPLE>\n" + example_prompt + "\n</EXAMPLE>"
            example_prompts.append(example_prompt)

        example_prompts = "\n\n".join(example_prompts)
        example_prompts += "\n\n"

        return example_prompts

    def get_conversation(
        self,
        messages: List[List[str]] = (),
        function_call: Dict = {},
        roles: List = [],
        colon: str = "",
        end_token: str = "",
        separators: List = [],
        predict: bool = True,
        add_generation_prompt: bool = False,
        add_fc_generation_prompt: bool = False,
    ) -> str:
        if not messages:
            return ""

        # if not override
        if not roles:
            roles = self.roles
        if not colon:
            colon = self.colon
        if not separators:
            separators = self.separators
        if not end_token:
            end_token = self.end_token

        user_role, assistant_role = roles

        # content window
        messages = messages[-(self.offset + 1) :]

        # exclude the system message
        contain_system = False
        if messages[0]["role"] == "system":
            messages = messages[1:]
            contain_system = True

        # construct the message prompt
        ret = ""
        for midx, message in enumerate(messages):
            if message["role"] == "user":
                user_content = message["content"]
                ret += f"{user_role}{colon}{user_content}{end_token}{separators[1]}"
            elif message["role"] == "assistant":
                assistant_content = ""
                if "function_call" in message:
                    fn_call = message["function_call"]
                    fn_name = fn_call.get("name", fn_call.get("function"))
                    args = fn_call.get("parameters", fn_call.get("arguments"))
                    assert fn_name is not None and args is not None
                    if args == {}:
                        continue
                    if self.template_name == "toolace":
                        params_str = ", ".join(
                            f"{k}={repr(v)}" for k, v in args.items()
                        )
                        assistant_content += f"[{fn_name}({params_str})]"
                    elif self.template_name == "xlam":
                        call = {"name": fn_name, "arguments": args}
                        function_call_json = json.dumps([call], ensure_ascii=False)
                        assistant_content += function_call_json
                    elif self.template_name == "llama31":
                        call = {"name": fn_name, "parameters": args}
                        function_call_json = json.dumps(call, ensure_ascii=False)
                        assistant_content += (
                            self.function_call_prefix
                            + function_call_json
                            + self.function_call_suffix
                        )
                    else:  # qwen and etc
                        call = {
                            "name": fn_name,
                            "arguments": args,
                        }
                        function_call_json = json.dumps(call, ensure_ascii=False)
                        assistant_content += (
                            self.function_call_prefix
                            + function_call_json
                            + self.function_call_suffix
                        )
                # add content
                assistant_content += message["content"]
                ret += f"{assistant_role}{colon}{assistant_content}{end_token}"
                if midx + 1 < len(messages) or not predict:
                    ret += separators[2]
                if midx + 1 == len(messages) and predict:
                    ret = ret.strip()

        # add the prefix to trigger the arguments output
        if function_call:
            function_name = function_call["name"]
            assistant_content = f'{{"name": "{function_name}", "arguments":'
            ret += (
                assistant_role + colon + self.function_call_prefix + assistant_content
            )
        if add_generation_prompt:
            ret += colon + assistant_role
        if add_fc_generation_prompt:
            ret += self.function_call_prefix

        return ret

    def get_functions(self, functions):
        if self.template_name == "qwen":
            prompt = "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
            for function in functions:
                prompt += json.dumps(function) + "\n"
            prompt += "</tools>"
            prompt += (
                "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                + '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
            )
        elif self.template_name == "qwen3":
            prompt = "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
            for function in functions:
                prompt += json.dumps(function) + "\n"
            prompt += "</tools>"
            prompt += (
                "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                + '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
            )
        elif self.template_name == "qwen3-think":
            prompt = "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
            for function in functions:
                prompt += json.dumps(function) + "\n"
            prompt += "</tools>"
            prompt += (
                "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                + '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
            )
        elif self.template_name == "xlam":
            prompt = (
                "You have access to a set of tools. When using tools, make calls in a single JSON array:\n"
                + '[{"name": "tool_call_name", "arguments": {"arg1": "value1", "arg2": "value2"}}, ... (additional parallel tool calls as needed)]'
            )
            prompt += "The available tools are:"
            for function in functions:
                prompt += json.dumps(function, indent=4) + "\n"
            prompt += 'To call a function with a JSON object of the following format: {"name": "function_name", "parameters": dictionary of argument name and its value (e.g. {"param1": "param1_value", "param2": "param2_value"})}'
        elif self.template_name == "llama31":
            prompt = "You have access to the following functions. To call a function, please respond with JSON for a function call."
            prompt += '\nRespond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.\n'
            for function in functions:
                prompt += json.dumps(function, indent=4) + "\n"
            prompt += "\n"
        elif self.template_name == "toolace":
            prompt = "You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose"
            prompt += "If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]"
            prompt += (
                "Here is a list of functions in JSON format that you can invoke.\n"
            )
            prompt += json.dumps(functions) + "\n"
        else:
            raise Exception("No format for function")

        return prompt

    def get_response(
        self,
        text,
        function_call={},
        required=["function_call", "content"],
        stop_strs=[("</s>", 0), ("\n\n", 0)],
    ):
        def extract_first_dict(s):
            first_left_brace = s.find("{")
            start = first_left_brace
            while start >= 0 and start <= len(s):
                index = s.find("}", start)
                if index == -1:  # No more occurrences
                    break
                try:
                    json_obj = s[first_left_brace : index + 1]
                    json_dict = json.loads(json_obj)
                    json_str = json.dumps(json_dict)
                    return json_str
                except:
                    start = index + 1
            return ""

        text = text.lower().strip().replace("\_", "_")
        text = " ".join(text.split())
        text = " ".join(text.splitlines())

        function_call_suffix = self.function_call_suffix.strip()
        function_call_prefix = self.function_call_prefix.strip()

        if "name" in function_call and "arguments" not in function_call:
            function_name = function_call["name"]
            response_prefix = f'{{"function": "{function_name}", "arguments": '
            text = response_prefix + text

        parsed_function_call = {}
        parsed_response = text

        if "function_call" in required:
            try:
                if "{" in text:  # TODO: HOTFIXED
                    parsed_function_call = extract_first_dict(text)
                    parsed_response = text.replace(parsed_function_call, "")
                    parsed_function_call = json.loads(parsed_function_call)

                    # if function_call_suffix in text or "{" in text:
                    #     parsed_response = text.split(function_call_suffix)[1].strip()
                    #     parsed_function_call = text.split(function_call_suffix)[0].strip()
                    #     if function_call_prefix in parsed_function_call:
                    #         parsed_function_call = parsed_function_call.split(function_call_prefix)[1].strip()
                    #     parsed_function_call = json.loads(parsed_function_call)
                    # else:
                    #     parsed_function_call = extract_first_dict(text)
                    #     parsed_response = text.replace(parsed_function_call, "")
                    #     parsed_function_call = json.loads(parsed_function_call)

                    # HOTFIX for parameters -> arguments
                    if (
                        "parameters" in parsed_function_call
                        and "arguments" not in parsed_function_call
                    ):
                        parsed_function_call["arguments"] = parsed_function_call.pop(
                            "parameters"
                        )

            except Exception as error:
                print(error)
                parsed_function_call = {}

        # NEW FIX: If parsed_function_call is empty, try parsing content
        if not parsed_function_call and parsed_response.startswith("{"):
            try:
                parsed_function_call = json.loads(parsed_response)
                parsed_response = ""

                if "parameters" in parsed_function_call:
                    parsed_function_call["arguments"] = parsed_function_call.pop(
                        "parameters"
                    )

            except Exception as error:
                print("Second parsing attempt failed:", error)
                parsed_function_call = {}

        if "content" in required:
            if function_call_prefix in parsed_response:
                parsed_response = parsed_response.split(function_call_prefix)[0].strip()

        for s in stop_strs:
            if s[0] in parsed_response:
                parsed_response = parsed_response.split(s[0])[s[1]]

        ret = {
            "role": "assistant",
            "content": parsed_response,
            "function_call": parsed_function_call,
        }
        return ret
