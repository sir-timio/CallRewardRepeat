# from chatbots.base_handler import BaseHandler
# import json
# from .utils import extract_first_dict
# import re


# class HammerHandler(BaseHandler):
#     def format_prompt(self, messages, functions=None, function_call=None):
#         if function_call:
#             print('do not supported yet')
        
#         formatted_prompt = self.tokenizer.apply_chat_template(
#             messages, 
#             tools=functions, 
#             tokenize=False, 
#             add_generation_prompt=True
#         )
#         return formatted_prompt

#     def parse_response(self, text, required: list[str] = ['function_call', 'content']):
#         text = text.lower().strip()
#         text = text.replace("\_", "_").replace("```", "").replace("'", '"')
        
#         result = {
#             "content": "",
#             "function_call": {}
#         }

#         if "function_call" in required:
#             try:
#                 response_json = json.loads(text)
#                 if isinstance(response_json, list) and len(response_json):
#                     response_json = response_json[0]
#                 if isinstance(response_json, dict) and ("name" in response_json or "function" in response_json):
#                     result["function_call"]["name"] = response_json.get("name", response_json.get("function"))
#                     result["function_call"]["arguments"] = response_json.get("parameters", response_json.get("arguments", {}))
#                 else:
#                     result["content"] = text
#             except json.JSONDecodeError:
#                 extracted_json = extract_first_dict(text)
#                 if extracted_json:
#                     function_call = json.loads(extracted_json)
#                     result["function_call"]["name"] = function_call.get("name", function_call.get("function"))
#                     result["function_call"]["arguments"] = function_call.get("parameters", function_call.get("arguments", {}))
#                     result["content"] = text.replace(extracted_json, "").strip()
#                 else:
#                     result["content"] = text
#         else:
#             result["content"] = text

#         return result