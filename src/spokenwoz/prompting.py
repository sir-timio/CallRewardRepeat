import os
import json
import random


import argparse
import logging
import yaml


from src.utils import str2bool, str2list, string2int, tod_instructions, tod_notes
from chatbots.utils import *
from chatbots.conversation import Conversation
from chatbots.llm import *
from src.multiwoz.schema2function import schema2function
from src.spokenwoz.utils import get_data_split


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

domain2function_mapping = {
    "hotel": "find_book_hotel",
    "train": "find_book_train",
    "attraction": "find_attraction",
    "restaurant": "find_book_restaurant",
    "taxi": "book_taxi",
    "profile": "fill_profile",
    "hospital": "find_hospital",
    "police": "police",
}


def load_schema(path="data/spokenwoz/normalized_schema.yaml"):
    with open(path, "r") as stream:
        try:
            scheme = yaml.safe_load(stream)
            return scheme
        except yaml.YAMLError as exc:
            print(exc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "dev", "test"]
    )  #
    parser.add_argument(
        "--n_sample", type=int, default=80, help="number of evaluated dialogues"
    )  #
    parser.add_argument(
        "--template",
        type=str,
        default="llama2",
        help="the conversation template of data",
    )  #
    parser.add_argument(
        "--all_turn",
        type=str2bool,
        default=False,
        help="if add all turns of function calls",
    )  #

    parser.add_argument(
        "--all_funcs",
        type=str2bool,
        default=False,
        help="if add all possible functions into system prompt",
    )  #

    args, unknown = parser.parse_known_args()
    print(args)

    # load schema, examples, data
    train_data, val_data, test_data = get_data_split(
        n_train=args.n_sample,
        n_val=args.n_sample,
        n_test=args.n_sample,
        return_list=False,
    )
    schema = load_schema()

    if args.split == "train":
        data = train_data
    elif args.split == "dev":
        data = val_data
    elif args.split == "test":
        data = test_data

    # save data path
    data_path = f"./data/pre-training_corpora/prompting_data/spokenwoz_{args.n_sample}/"
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    save_path = f"{data_path}/{args.split}-{args.template}-allturn{args.all_turn}-allfuncs{args.all_funcs}.json"

    conversation = Conversation(
        template_name=args.template,
        function_type="json",
    )

    if not args.all_funcs:
        raise NotImplementedError()

    # all the processed dials for each task
    processed_data = []

    # {'dial_id': 'MUL0003',
    #  'turn_num': 7,
    #  'user': 'yes , of course my name is rohanna newberg .',
    #  'resp': '',
    #  'nodelx_resp': 'casual name .',
    #  'bspn_dict': {'[profile]': {'name': 'wilhelmina newberg'},
    #   '[restaurant]': {'area': 'centre',
    #    'food': 'indian',
    #    'day': 'friday',
    #    'people': '1',
    #    'time': '14:00'}},
    #  'bspn': '[profile] name wilhelmina newberg [restaurant] area centre day friday food indian people 1 time 14:00',
    #  'turn_bspn_dict': {'[profile]': {'name': 'wilhelmina newberg'}},
    #  'turn_bspn': '[profile] name wilhelmina newberg',
    #  'bsdx': '',
    #  'db': 0,
    #  'dspn': '[profile]',
    #  'aspn': '',
    #  'aspn_dict': {},
    #  'all_domains': ['[hotel]', '[restaurant]', '[taxi]', '[profile]']}
    all_functions = []
    # for domain in ["[taxi]", "[train]", "[attraction]", "[hotel]", "[restaurant]"]:
    for service in schema:
        function = schema2function(service, rename_mapping=domain2function_mapping)
        all_functions.append(function)

    # prepare input and output for each task
    for dial_id, turns in data.items():
        messages = []
        # system instruction
        system_messages = [random.choice(tod_instructions)]
        system_messages.extend(tod_notes)
        system_message = "\n".join(system_messages)
        messages.append({"role": "system", "content": system_message})

        # add conversation messages
        for turn in turns:
            usr = turn["user"]
            resp = turn["nodelx_resp"]
            messages.append({"role": "user", "content": usr})

            turn_domain = turn["dspn"]
            turn_bs_dict = turn["turn_bspn_dict"]

            bs_dict = turn["bspn_dict"]

            function_call_dict = {}
            # all turn for spoken woz means nothing due to lack of info in original dataset
            if args.all_turn:  # add function call at all the turns
                if turn_domain in bs_dict:
                    function_call_dict = {
                        "name": domain2function_mapping[turn_domain[1:-1]],
                        "parameters": bs_dict[turn_domain],
                    }
            else:  # only add function call when there are update
                if turn_domain in turn_bs_dict:
                    function_call_dict = {
                        "name": domain2function_mapping[turn_domain[1:-1]],
                        "parameters": bs_dict[turn_domain],
                    }

            if function_call_dict:
                messages.append(
                    {
                        "role": "assistant",
                        "content": resp,
                        "function_call": function_call_dict,
                    }
                )
            else:
                messages.append({"role": "assistant", "content": resp})

        # construct the prompt, exclude the conversation part
        system_prompt = conversation.get_prompt(
            system_message=system_message,
            # functions=all_functions if args.all_funcs else functions,
            functions=all_functions if args.all_funcs else None,
            messages=[],
        )

        # construct each turn of the conversation with function calls
        conversation_prompt = []
        for message in messages[1:]:
            turn_prompt = conversation.get_conversation([message], predict=False)
            conversation_prompt.append(
                {"role": message["role"], "content": turn_prompt}
            )

        processed_data.append(
            {
                "system": system_prompt,
                # "functions": all_functions if args.all_funcs else functions,
                "functions": all_functions if args.all_funcs else None,
                "conversation": conversation_prompt,
                "dial_id": dial_id,
            }
        )

    # summarize
    print(f"Total dialogues: {len(data)}!")
    print(f"Total samples: {len(processed_data)}")

    # save data
    with open(save_path, "w") as file:
        json.dump(processed_data, file, indent=4)
