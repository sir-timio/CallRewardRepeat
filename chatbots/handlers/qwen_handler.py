from chatbots.base_handler import BaseHandler
import json
from .utils import extract_first_dict
import re

class QwenHandler(BaseHandler):
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
            )
        # prompt += "<tool_call>"
        # print(f'Prompt: {prompt[:100]} \n...\n {prompt[-100:]}')
        return prompt 
    

    def parse_response(self, text, required: list[str] = ['function_call', 'content']):
        # text = "<tool_call>{" + text
        print(f'Before parsing: {text}')
        text = text.lower().strip()
        text = text.replace("\_", "_").replace("\n", " ").replace("```", "")
        fc_prefix = "<tool_call>"
        fc_suffix = "</tool_call>"
        
        result = {
            "content": "",
            "function_call": {}
        }

        if "function_call" in required:
            try:
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
                    response_json = json.loads(extracted_json)
                    if "name" in response_json or "function" in response_json:
                        result["function_call"]["name"] = response_json.get("name", response_json.get("function"))
                        result["function_call"]["arguments"] = response_json.get("parameters", response_json.get("arguments", {}))
                    else:
                        result["function_call"]["name"] = list(response_json.keys())[0]
                        result["function_call"]["arguments"] = list(response_json.values())[0]
                else:
                    result["content"] = text
            except Exception as e:
                print(f'Handler error: {e}')
                print(f'Text: {text}')
        else:
            result["content"] = text

        return result