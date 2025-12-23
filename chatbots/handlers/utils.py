import re
import json

def split_function_nl(assistant_output, fc_prefix, fc_suffix):
    """takes only the very first function"""
    prefix_pos = assistant_output.find(fc_prefix)
    
    if prefix_pos == -1:
        return "", assistant_output
    
    search_start = prefix_pos + len(fc_prefix)
    suffix_pos = assistant_output.find(fc_suffix, search_start)
    
    if suffix_pos == -1:
        return "", assistant_output
    
    suffix_end = suffix_pos + len(fc_suffix)
    function_call = assistant_output[prefix_pos:suffix_end]
    
    before_function = assistant_output[:prefix_pos]
    after_function = assistant_output[suffix_end:]
    response = before_function + after_function
    
    return function_call, response

# def split_function_nl(assistant_output, fc_prefix, fc_suffix):
        
#     if fc_prefix in assistant_output and fc_suffix in assistant_output:
#         prefix_start = assistant_output.find(fc_prefix)
#         suffix_start = assistant_output.rfind(fc_suffix)
        
#         if prefix_start != -1 and suffix_start != -1 and suffix_start > prefix_start:
#             suffix_end = suffix_start + len(fc_suffix)
#             function_call = assistant_output[prefix_start:suffix_end]
#             response = assistant_output[suffix_end:]
#         else:
#             function_call = ""
#             response = assistant_output
#     else:
#         function_call = ""
#         response = assistant_output
#     return function_call, response

def extract_first_dict(s):
    first_left_brace = s.find("{")
    start = s.find("{")
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