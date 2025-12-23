from chatbots.base_handler import BaseHandler
# import json
import orjson as json
from .utils import extract_first_dict, split_function_nl
import re

class TrainedModelHandler(BaseHandler):
    def format_prompt(self, messages, functions=None, function_call=None, add_generation_prompt=True):
        system_message = None
        for message in messages:
            if message['role'] == 'system':
                system_message = message['content']
                messages = messages[1:]
                break
        prompt = self.conversation.get_prompt(
            system_message=system_message,
            messages=messages,
            functions=functions,
            add_generation_prompt=add_generation_prompt,
            # add_fc_generation_prompt=True,
            add_fc_generation_prompt=False,
            )
        # prompt += "<tool_call>"
        # print(f'Prompt: {prompt[:100]} \n...\n {prompt[-100:]}')
        return prompt


    def parse_response(self, text, required: list[str] = ['function_call', 'content']):
        text = text.strip().lstrip(self.conversation.roles[1].strip()).strip()
        result = {
            "raw_content": text,
            "content": "",
            "function_call": {}
        }

        fc_prefix = self.conversation.function_call_prefix.strip()
        fc_suffix = self.conversation.function_call_suffix.strip()

        # text = "<tool_call>{" + text
        if text.find(fc_prefix) * text.find(fc_suffix) < 0:
            text = fc_prefix + text

        verbose = False
        if verbose:
            print(f'Before parsing: {text[:200]}')
            # print(f'Before parsing: {split_function_nl(text, fc_prefix, fc_suffix)[0]}')
        text = text.lower().strip()
        text = text.replace("\_", "_").replace("\n", " ").replace("```", "")



        if "function_call" in required:
            if self.conversation.template_name == 'toolace':
                parsed_result = self._parse_toolace_format(text, fc_prefix, fc_suffix)
                if parsed_result:
                    return parsed_result
            try:
                if fc_prefix in text and fc_suffix in text:
                    text = split_function_nl(text, fc_prefix, fc_suffix)[0]

                if fc_prefix in text:
                    text = text.split(fc_prefix)[1].strip()
                if fc_suffix in text:
                    text = text.split(fc_suffix)[0].strip()
                response_json = json.loads(text)
                if isinstance(response_json, list):
                    response_json = response_json[0]
                if isinstance(response_json, dict) and ("name" in response_json or "function" in response_json):
                    result["function_call"]["name"] = response_json.get("name", response_json.get("function"))
                    result["function_call"]["arguments"] = response_json.get("parameters", response_json.get("arguments", {}))
                else:
                    result["function_call"]["name"] = list(response_json.keys())[0]
                    result["function_call"]["arguments"] = list(response_json.values())[0]

            except json.JSONDecodeError:
                extracted_json = extract_first_dict(text)
                if extracted_json:
                    try:
                        response_json = json.loads(extracted_json)
                        if "name" in response_json or "function" in response_json:
                            result["function_call"]["name"] = response_json.get("name", response_json.get("function"))
                            result["function_call"]["arguments"] = response_json.get("parameters", response_json.get("arguments", {}))
                        else:
                            try:
                                result["function_call"]["name"] = list(response_json.keys())[0]
                                result["function_call"]["arguments"] = list(response_json.values())[0]
                            except Exception:
                                result["content"] = text
                    except Exception as e:
                        result["content"] = text
                        print(f'Handler error: {e}')
                        print(f'Text: {text}')
                else:
                    result["content"] = text
            except Exception as e:
                print(f'Handler error: {e}')
                print(f'Text: {text}')
        else:
            result["content"] = text

        if verbose:
            print(f'result: {result}')

        return result

    def _parse_toolace_format(self, text, fc_prefix, fc_suffix):
        """
        Parsing for toolace: [function_name(param1='value1', param2='value2')]
        """
        try:
            # # Extract text between prefix and suffix
            # text = split_function_nl(text, fc_prefix, fc_suffix)
            # if fc_prefix in text:
            #     text = text.split(fc_prefix).strip()
            # if fc_suffix in text:
            #     text = text.split(fc_suffix).strip()

            # Look for function call in square brackets
            pattern = r'$$([a-zA-Z_][a-zA-Z0-9_]*)$$(.*?)$$$$'
            match = re.search(pattern, text)

            if match:
                function_name = match.group(1)
                params_str = match.group(2)

                # Parse params
                arguments = {}
                if params_str.strip():
                    # Split by commas, considering quotes
                    params = self._split_params(params_str)
                    for param in params:
                        if '=' in param:
                            key, value = param.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            # Remove quotes if exist
                            if (value.startswith("'") and value.endswith("'")) or \
                            (value.startswith('"') and value.endswith('"')):
                                value = value[1:-1]
                            arguments[key] = value

                return {
                    "content": "",
                    "function_call": {
                        "name": function_name,
                        "arguments": arguments
                    }
                }
        except Exception as e:
            print(f'Toolace parsing error: {e}')
            return None

        return None

    def _split_params(self, params_str):
        """
        Splits params string by commas, considering quotes.
        """
        params = []
        current_param = ""
        in_quotes = False
        quote_char = None

        for char in params_str:
            if char in ["'", '"'] and not in_quotes:
                in_quotes = True
                quote_char = char
                current_param += char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
                current_param += char
            elif char == ',' and not in_quotes:
                if current_param.strip():
                    params.append(current_param.strip())
                current_param = ""
            else:
                current_param += char

        if current_param.strip():
            params.append(current_param.strip())

        return params
