import json
import random
from collections import defaultdict
from src.utils import fc_prefix, fc_suffix, fc_prompt_prefix, fc_prompt_suffix, tod_instructions, tod_notes

def format_functions(functions, function_type="json"):
    """
    Formats a list of functions into a prompt string.

    Args:
        functions (list[dict]): List of function definitions.
        function_type (str): Output format type ("json" or "text").

    Returns:
        str: Formatted prompt string.
    """

    prompts = []

    if function_type == "json":
        for function in functions:
            function_dict = {
                "name": function["name"],
                "description": function["description"],
                "arguments": [],
            }

            parameters = function["parameters"]["properties"]
            for param_name, param_info in parameters.items():
                param_dict = {
                    "name": param_name,
                    "type": param_info["type"],
                    "description": param_info["description"],
                }
                if "enum" in param_info:
                    param_dict["possible_values"] = param_info["enum"]
                function_dict["arguments"].append(param_dict)

            if "required" in function["parameters"]:
                function_dict["required"] = function["parameters"]["required"]

            function_prompt = fc_prefix + json.dumps(function_dict, indent=4) + fc_suffix
            prompts.append(function_prompt)

        prompt = fc_prompt_prefix + "\n".join(prompts) + fc_prompt_suffix

    elif function_type == "text":
        for function in functions:
            lines = [
                f"Function name: {function['name']}",
                f"Function description: {function['description']}",
                "Function arguments:",
            ]

            parameters = function["parameters"]["properties"]
            for param_name, param_info in parameters.items():
                param_desc = f" - {param_name} ({param_info['type']}): {param_info['description']}"
                if "enum" in param_info:
                    enum_values = ", ".join(param_info["enum"])
                    param_desc += f" (must be one of {enum_values})"
                lines.append(param_desc)

            prompts.append("\n".join(lines))

        prompt = fc_prompt_prefix + "\n\n".join(prompts) + fc_prompt_suffix

    else:
        raise NotImplementedError("Unsupported function_type")

    prompt += ('To call a function (tool) with a JSON object of the following format: '
               '{"function": "function_name", "arguments": {"argument1": "argument_value", '
               '"argument2": "argument_value"}}')

    return prompt


def create_corpus(dialogues, functions, fc_prefix=fc_prefix, fc_suffix=fc_suffix):
    """
    Creates a structured corpus dataset from dialogues and functions.

    Args:
        dialogues (dict): Dictionary containing dialogue data.
        functions (list[dict]): List of available functions.
        fc_prefix (str): Prefix string for formatting function calls.
        fc_suffix (str): Suffix string for formatting function calls.

    Returns:
        list[dict]: Structured dataset ready for serialization.
    """
    dataset = []

    system_message = random.choice(tod_instructions)

    system_message += format_functions(functions)

    for dialog_id, dialog in dialogues.items():
        dialog_data = {
            "system_message": system_message,
            "conversation": [],
            "dial_id": dialog_id,
            "domains": [],
        }

        sorted_dialogue = sorted(dialog, key=lambda x: x['turn_num'])

        for turn in sorted_dialogue:
            dialog_data["conversation"].append({
                "role": "user",
                "content": turn["user"]
            })

            function_call_json = json.dumps(turn["turn_bspn_dict"], ensure_ascii=False)
            assistant_content = f'{fc_prefix}{function_call_json}{fc_suffix}{turn["nodelx_resp"]}'

            dialog_data["conversation"].append({
                "role": "assistant",
                "content": assistant_content
            })

        if "all_domains" in sorted_dialogue[-1]:
            dialog_data["domains"] = sorted_dialogue[-1]["all_domains"]

        dataset.append(dialog_data)

    return dataset

# Example usage:
# train_dataset = create_corpus(dialogues, functions)

# Save dataset to file
# with open('data/multiwoz/prompted_corpus/train.json', 'w', encoding='utf-8') as f:
#     json.dump(train_dataset, f, ensure_ascii=False, indent=2)
