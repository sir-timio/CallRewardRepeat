
def schema2function(service, rename_mapping={}):
    args_alias = "parameters"
    type_alias = "dict"

    # if template in ['toolace',]:
    #     args_alias = 'arguments'
    # else:
    #     args_alias = 'parameters'

    # if template in ['toolace',]:
    #     type_alias = 'dict'
    # else:
    #     type_alias = 'object'

    function = {
        "name": "",
        "description": "",
        args_alias: {"type": type_alias, "properties": {}, "required": []},
    }
    service_name = service["service_name"]
    function["name"] = rename_mapping.get(service_name, service_name)

    contain_name_parameter = False
    possible_integers = set(list(map(str, range(10))))
    possible_bools = set(["yes", "no"])

    for slot in service["slots"]:
        if slot.get("is_informable", None):
            slot_name = slot["name"].split("-")[-1].strip()
            parameter = {}
            parameter["description"] = slot["description"]
            parameter["type"] = "string"
            if "possible_values" in slot:
                if any([v in slot["possible_values"] for v in possible_bools]):
                    parameter["type"] = "boolean"
                elif any([v in slot["possible_values"] for v in possible_integers]):
                    parameter["type"] = "integer"
            if slot_name in ["arrive", "leave", "time"]:
                parameter["type"] = "time (HH:mm)"
                # if template != "chatgpt":
                #     parameter["type"] = "time (HH:mm)"

            if "possible_values" in slot:
                if slot["is_categorical"]:
                    parameter["enum"] = [str(v) for v in slot["possible_values"]]
                else:
                    if slot["possible_values"]:
                        examples = ", ".join(slot["possible_values"][:10])
                        parameter["description"] += f" such as {examples}, etc."

            if slot_name == "name":
                contain_name_parameter = True

            function[args_alias]["properties"][slot_name] = parameter

    function[args_alias]["required"] = service["intents"][0]["required_slots"]

    description = service["description"] + ". "
    CAUTIONS = [
        "Set the value as : 'dontcare' ONLY when the user EXPLICITLY states they have no specific preference for a parameter.",
    ]
    if contain_name_parameter:
        CAUTIONS.append(
            "Always record the exact value of the 'name' parameter when mentioned. Avoid using pronouns or coreferences like 'the hotel' or 'the restaurant.'"
        )
    description += " ".join(CAUTIONS)
    function["description"] = description

    return function
