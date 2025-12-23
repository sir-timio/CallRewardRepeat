import logging
import os
import random
import shutil
import sys
from copy import deepcopy
from datetime import datetime
from tempfile import tempdir
from uuid import uuid4

import fire
import orjson as json
import pandas as pd
import torch
from clearml import Dataset, InputModel, OutputModel, Task
from tqdm import tqdm

from clearml_utils import create_task, execute_remotely, update_data_path, copy_data

if torch.cuda.is_available():
    os.system(f"{sys.executable} -m spacy download en_core_web_sm")
    import torch
    import unsloth
    from unsloth import FastLanguageModel

    from src.multiwoz.postprocess import unzip_session_data
    from chatbots.utils import *
    from chatbots.llm import chat_completion
    from chatbots.llm import *
    from src.multiwoz.utils import *
    from src.multiwoz.utils.config import *
    from src.multiwoz.utils.reader import *
    from src.multiwoz.utils.utils import paser_bs_to_dict
    from src.multiwoz.utils.compute_joint_acc import compute_jacc, zip_result

from clearml import Dataset, InputModel, OutputModel, Task
from huggingface_hub import login
from transformers import AutoModelForCausalLM, set_seed

from src.utils import *

login(os.getenv("HF_TOKEN"))

set_seed(42)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
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
function2domain = {v: k for k, v in domain2function_mapping.items()}

FIND_DOMAINS = {"hotel", "train", "attraction", "restaurant", "hospital"}
BOOK_DOMAINS = {"taxi"}
FILL_DOMAINS = {"profile"}


def pprint_metrics(df):
    renaming = {
        "DST_Joint_Accuracy": "JGA",
        "DST_Slot_F1": "F1",
        "DST_Precision": "precision",
        "DST_Recall": "recall",
    }
    df.rename(renaming, axis=1, inplace=True)

    cols = df.columns
    new_cols = []
    for c in cols:
        new_cols.append(
            c.replace("DST_Joint_Accuracy_per_", "JGA_").replace(
                "DST_Slot_F1_per_", "F1_"
            )
        )
    df.columns = new_cols
    df["JGA_avg"] = df[[col for col in df.columns if col.startswith("JGA_")]].mean(
        axis=1
    )

    df["F1_avg"] = df[[col for col in df.columns if col.startswith("F1_")]].mean(axis=1)
    print(df[["JGA_avg", "F1_avg", "JGA", "F1", "F1_avg"]])
    return df[["JGA_avg", "F1_avg", "JGA", "F1", "F1_avg"]]


def reload_model(path):
    return AutoModelForCausalLM.from_pretrained(
        path,
        return_dict=True,
        torch_dtype=torch.float16,
        **{"device_map": "auto"},
    )


def normalize_domain(domain, schema):
    d = domain.strip("[]")
    for s in schema:
        if s["service_name"].lower() == d.lower():
            return f"[{s['service_name']}]"
    return f"[{d}]"


def process_assistant_response(
    assistant_message,
    user_goal,
    schema,
    domains,
    schema2function,
    domain2function_mapping,
    function2domain,
):
    function_call = assistant_message["function_call"]

    if not function_call:
        return None

    # step 1: get the domain

    if "name" in function_call:
        try:
            pred_function = function_call["name"].strip().lower()
        except:
            return None
    elif "function" in function_call:
        try:
            pred_function = function_call["function"].strip().lower()
        except:
            return None
    else:
        return None

    turn_domain = function2domain.get(pred_function, pred_function)
    turn_domain = f"[{turn_domain}]"

    current_function = None
    # step 2: get the current function
    for service in schema:
        if service["service_name"].lower() == turn_domain[1:-1].lower():
            current_function = schema2function(
                service,
                rename_mapping=domain2function_mapping,
            )
    if current_function is None:
        return None

    turn_bs_dict_gen = function_call["arguments"]
    if isinstance(turn_bs_dict_gen, str):
        try:
            turn_bs_dict_gen = json.loads(turn_bs_dict_gen)
        except:
            try:
                turn_bs_dict_gen = json.loads(turn_bs_dict_gen.replace("'", '"'))
            except:
                return None
    if not isinstance(turn_bs_dict_gen, dict):
        print(f"not dict! {turn_bs_dict_gen}")
        turn_bs_dict_gen = {}

    # update user goal
    if turn_domain in domains:
        if turn_domain not in user_goal:
            user_goal[turn_domain] = {}

        # clean the generation and update user goal
        for slot, value in turn_bs_dict_gen.items():
            slot = slot.strip().lower()
            value = str(value).strip().lower()
            # only update the valid generations
            if slot in current_function["parameters"]["properties"]:
                if "enum" not in current_function["parameters"]["properties"][slot]:
                    user_goal[turn_domain][slot] = value
                else:
                    possible_values = set(
                        [
                            v.lower()
                            for v in current_function["parameters"]["properties"][slot][
                                "enum"
                            ]
                        ]
                    )
                    if value in possible_values:
                        user_goal[turn_domain][slot] = value
        user_goal = {k.lower(): v for k, v in user_goal.items()}
        return turn_domain, user_goal


def build_messages(eval_turns, turn_idx, eval_turn, domain2function_mapping):
    messages = []
    system_messages = [random.choice(tod_instructions)]
    system_messages.extend(tod_notes)
    system_message = "\n".join(system_messages)
    messages.append({"role": "system", "content": system_message})

    for prev_idx in range(turn_idx):
        prev_turn = eval_turns[prev_idx]
        usr = prev_turn["user"]
        resp = prev_turn["nodelx_resp"]
        prev_domain = prev_turn["dspn"]
        prev_bs_dict = prev_turn["bspn_dict_gen"]
        messages.append({"role": "user", "content": usr})
        assistant_message = {"role": "assistant", "content": resp}

        assistant_message = {"role": "assistant", "content": prev_turn["nodelx_resp"]}
        if prev_bs_dict:
            if prev_domain in prev_bs_dict:
                function_call_dict = {
                    "name": domain2function_mapping.get(prev_domain[1:-1], prev_domain),
                    "parameters": prev_bs_dict[prev_domain],
                }
                assistant_message["function_call"] = function_call_dict
        messages.append(assistant_message)
    messages.append({"role": "user", "content": eval_turn["user"]})
    return messages


def prepare_evaluation(data):
    eval_data = {}
    for dial_id, turns in data.items():
        eval_turns = []
        for turn in turns:
            eval_turn = deepcopy(turn)
            for key in [
                "dial_id",
                "turn_num",
                "user",
                "resp",
                "nodelx_resp",
                "resp_gen",
                "dspn",
                "dspn_gen",
                "bsdx",
                "bspn",
                "bspn_gen",
                "bspn_dict",
                "bspn_dict_gen",
                "turn_bspn",
                "turn_bspn_gen",
                "turn_bspn_dict",
                "turn_bspn_dict_gen",
                "db",
                "db_gen",
                "aspn",
                "aspn_gen",
                "aspn_dict",
                "aspn_dict_gen",
                "all_domains",
            ]:
                if key in turn:
                    eval_turn[key] = turn[key]
                elif "dict" in key:
                    eval_turn[key] = {}
                else:
                    eval_turn[key] = ""
            eval_turns.append(eval_turn)
        eval_data[dial_id] = eval_turns
    return eval_data


def evaluate(
    dataset: str,
    split: str = "test",
    n_eval: int = 100,
    ref_domain: bool = True,
    ref_bs: bool = False,
    multi_domain: bool = False,
    add_prev: bool = True,
    function_type: str = "json",  # ["json", "text"]
    dst_nshot: int = 0,
    nlg_nshot: int = 0,
    template: str = "",
    model: str = "gpt-3.5-0125",
    clearml_id: str = "null",
    name: str = "",
    # sft_clearml_id: str = "",
    # grpo_clearml_id: str = "",
    handler: str = "",
    chat_template: str = "",
    queue: str = "",
    temperature: float = 0.3,
    top_p: float = 0.2,
    generate: bool = False,
    debug: bool = False,
    verbose: bool = False,
    eval_valid: bool = False,
    task: Task = None,
    gpu_memory_utilization: float = 0.8,
    save_dir: str = None,
    save_path:str = None,
    reduce_inference_mem: bool = False,
):
    if template == "null":
        left, _ = name.split("_")
        template, _, dataset = left.split("-")
        dataset = dataset.strip()
        template = template.strip()
        print(f"TEMPLATE: {template}")
        print(f"DATASET: {dataset}")

    if queue:
        task_name = f"inference_{name}_{model}_id_{clearml_id}_{dataset}"
        task = create_task(task_name)
        execute_remotely(task)

    copy_data()


    print(f"Task: {task} {type(task)}")

    if not save_path:
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/{str(uuid4())}"
            print(f"SAVING PATH: {save_path}")
            os.makedirs(save_path, exist_ok=True)
        else:
            save_path = None

    print(f"Loading base model: {model}")
    if clearml_id != "null":
        model = InputModel(clearml_id).get_local_copy()

    default_cleaning_flag = False
    if dataset == "mwoz22":
        dataset_version = "2.2"
        default_cleaning_flag = True
        from src.multiwoz.postprocess import (get_data_split, load_examples,
                                              load_schema)
        from src.multiwoz.schema2function import schema2function

        data_prefix = "./data/multiwoz/data/"
        cfg = Config22(data_prefix)
        reader = MultiWozReader(tokenizer=None, cfg=cfg, data_mode=split)
        # load schema, examples, data
        train_data, val_data, test_data = get_data_split(
            dataset_version=dataset_version,
            reader=reader,
            n_train=-1,
            n_val=-1,
            n_test=n_eval,
            return_list=False,
        )
        schema = load_schema(dataset_version)
        examples = []
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
        function2domain = {v: k for k, v in domain2function_mapping.items()}
    elif dataset == "mwoz21":
        dataset_version = "2.1"
        default_cleaning_flag = True
        from src.multiwoz.postprocess import (get_data_split, load_examples,
                                              load_schema)
        from src.multiwoz.schema2function import schema2function

        data_prefix = "./data/multiwoz/data/"
        cfg = Config21(data_prefix)
        reader = MultiWozReader(tokenizer=None, cfg=cfg, data_mode=split)
        # load schema, examples, data
        train_data, val_data, test_data = get_data_split(
            dataset_version=dataset_version,
            reader=reader,
            n_train=-1,
            n_val=-1,
            n_test=n_eval,
            return_list=False,
        )
        schema = load_schema(dataset_version)
        examples = []
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
        function2domain = {v: k for k, v in domain2function_mapping.items()}
    elif dataset == "mwoz24":
        dataset_version = "2.4"
        default_cleaning_flag = True
        from src.multiwoz.postprocess import get_data_split, load_schema
        from src.multiwoz.schema2function import schema2function

        data_prefix = "./data/multiwoz/data/"
        cfg = Config24(data_prefix)
        reader = MultiWozReader(tokenizer=None, cfg=cfg, data_mode=split)
        # load schema, examples, data
        train_data, val_data, test_data = get_data_split(
            dataset_version=dataset_version,
            reader=reader,
            n_train=-1,
            n_val=-1,
            n_test=n_eval,
            return_list=False,
        )
        schema = load_schema(dataset_version)
        examples = []
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
        function2domain = {v: k for k, v in domain2function_mapping.items()}
    elif dataset == "swoz":
        from src.spokenwoz.prompting import load_schema
        from src.spokenwoz.utils import get_data_split

        train_data, val_data, test_data = get_data_split(
            n_train=-1,
            n_val=-1,
            n_test=n_eval,
            return_list=False,
        )
        schema = load_schema()
        examples = []
    elif dataset == "sgd":
        from src.sgd.postprocess import get_data_split, load_schema
        from src.sgd.schema2function import schema2function

        train_data, val_data, test_data = get_data_split(
            n_train=-1,
            n_val=n_eval,
            n_test=n_eval,
            return_list=False,
        )
        schema = load_schema()
        examples = []
        service_names = [s["service_name"] for s in schema]
        domain2function_mapping = {k: k for k in service_names}
        function2domain = {v: k for k, v in domain2function_mapping.items()}

    if split == "val":
        eval_data = val_data
    elif split == "test":
        eval_data = test_data
    eval_data = prepare_evaluation(eval_data)
    print(f"EVAL DATA SIZE: {len(eval_data)}")
    domains = set()
    for dial in eval_data.values():
        for turn in dial:
            if "bspn_dict" in turn:
                for d in turn["bspn_dict"].keys():
                    domains.add(d.lower())
    domains = sorted(list(domains))
    print("Domains for metric:", domains)

    # save data path
    data_prefix = f"./outputs/{dataset}"
    if not os.path.exists(data_prefix):
        os.makedirs(data_prefix, exist_ok=True)

    # save dst result path
    dst_eval_result_path = f"{data_prefix}/{split}{n_eval}-multi{multi_domain}-ref{ref_domain}-prev{add_prev}-{function_type}-dst{dst_nshot}shot-{model.replace('/', '#')}.json"
    dst_eval_error_path = f"{data_prefix}/{split}{n_eval}-multi{multi_domain}-ref{ref_domain}-prev{add_prev}-{function_type}-dst{dst_nshot}shot-{model.replace('/', '#')}-errors.json"

    task_name = dst_eval_result_path.rstrip(".json")

    evaluated_data = {}
    eval_result_path = dst_eval_result_path
    if os.path.exists(dst_eval_result_path):
        with open(dst_eval_result_path, "r") as file:
            try:
                evaluated_data = json.load(file)
            except Exception as e:
                print(f"{e}")
                evaluated_data = {}
            # finally:
            #     evaluated_data = {}
    elif not generate:
        raise NotImplementedError
    else:
        evaluated_data = {}

    # add the data that has been evaluated
    for dial_id, dp in evaluated_data.items():
        eval_data[dial_id] = dp

    # evaluation
    if generate:
        # load the model
        llm = chat_completion(
            model=model,
            template=template,
            handler=handler,
            # chat_template=chat_template,
            function_type=function_type,
            verbose=verbose,
            gpu_memory_utilization=gpu_memory_utilization,
            reduce_inference_mem=reduce_inference_mem
        )

        # dialogue_states = {
        #     dial_id: build_messages(turns, args)
        #     for dial_id, turns in eval_data.items()
        # }
        # dialogue_states = {dial_id: {'user_goal': {}, 'current_turn': 0} for dial_id in eval_data.keys()}

        dialogue_states = {
            dial_id: {"user_goal": {}, "current_turn": 0}
            for dial_id in eval_data.keys()
        }

        max_turns = max(len(turns) for turns in eval_data.values()) if eval_data else 0
        print(f"max turns: {max_turns}")
        for turn_idx in tqdm(range(max_turns), desc="Processing turns"):
            batch_data = []

            for dial_id, eval_turns in eval_data.items():
                if turn_idx < len(eval_turns):
                    eval_turn = eval_turns[turn_idx]
                    turn_domain = eval_turn["dspn"]
                    user_goal = dialogue_states[dial_id]["user_goal"]
                    if ref_bs:
                        user_goal = paser_bs_to_dict(eval_turn["bspn"])
                        eval_turn["bspn_gen"] = eval_turn["bspn"]
                        eval_turn["bspn_dict_gen"] = user_goal
                        dialogue_states[dial_id]["user_goal"] = user_goal
                        continue
                    # elif eval_turn.get("bspn_gen"):
                    #     user_goal = paser_bs_to_dict(eval_turn["bspn_gen"])
                    #     eval_turn["bspn_dict_gen"] = user_goal
                    #     dialogue_states[dial_id]["user_goal"] = user_goal
                    #     continue
                    else:  # collect batch
                        messages = build_messages(
                            eval_turns,
                            turn_idx,
                            eval_turn,
                            domain2function_mapping=domain2function_mapping,
                        )
                        functions = []

                        if ref_domain:
                            turn_domain = eval_turn.get("dspn", "[__]")
                            # if not turn_domain:
                            #     continue
                            for service in schema:
                                if service["service_name"] == "general":
                                    continue
                                elif (
                                    service["service_name"].lower()
                                    == turn_domain[1:-1].lower()
                                ):
                                    function = schema2function(
                                        service,
                                        rename_mapping=domain2function_mapping,
                                    )
                                    functions.append(function)
                        else:
                            raise NotImplementedError
                        batch_data.append(
                            {
                                "dial_id": dial_id,
                                "turn_idx": turn_idx,
                                "dspn": turn_domain,
                                "eval_turn": eval_turn,
                                "messages": messages,
                                "functions": functions,
                                "user_goal": deepcopy(user_goal),  # State copy
                            }
                        )

            if not batch_data:
                continue
            print(f"Turn {turn_idx}: processing {len(batch_data)} dialogues")

            batch_messages = [item["messages"] for item in batch_data]
            batch_functions = [item["functions"] for item in batch_data]

            chat_responses = llm.complete_batch(
                batch_messages=batch_messages,
                batch_functions=batch_functions,
                function_call={},
                required=["function_call"],
                temperature=temperature,
                top_p=top_p,
                max_tokens=128,
                n_seqs=1,
            )

            for i, item in enumerate(batch_data):
                dial_id = item["dial_id"]
                turn_idx = item["turn_idx"]
                eval_turn = item["eval_turn"]
                user_goal = item["user_goal"]
                assistant_message = chat_responses[i]
                eval_data[dial_id][turn_idx]["resp_gen"] = assistant_message
                result = process_assistant_response(
                    assistant_message=assistant_message,
                    user_goal=user_goal,
                    schema=schema,
                    schema2function=schema2function,
                    domains=domains,
                    domain2function_mapping=domain2function_mapping,
                    function2domain=function2domain,
                )
                if result:
                    turn_domain_gen, updated_user_goal = result
                    eval_data[dial_id][turn_idx]["bspn_dict_gen"] = deepcopy(
                        updated_user_goal
                    )

                    eval_data[dial_id][turn_idx]["dspn_gen"] = turn_domain_gen

                    dialogue_states[dial_id]["user_goal"] = updated_user_goal
                else:
                    eval_data[dial_id][turn_idx]["bspn_dict_gen"] = deepcopy(user_goal)

    """ Evaluations """
    eval_turns = unzip_session_data(eval_data)

    with open(eval_result_path, "w") as file:
        json.dump(eval_data, file, indent=4)

    # task.upload_artifact(
    #     "prediction",
    #     eval_result_path,
    # )

    """
    DST evaluation
    """
    print("=" * 25 + f" DST Evaluation " + "=" * 25)

    valid_turns = []
    for data in eval_turns:
        if data["bspn_gen"]:
            valid_turns.append(data)
    valid, total = len(valid_turns), len(eval_turns)
    if eval_valid:
        print("using valid turns")
        evaluated_turns = valid_turns
    else:
        evaluated_turns = eval_turns

    all_dev_result = zip_result(evaluated_turns)
    (
        dst_joint_acc,
        dev_f1,
        dev_precision,
        dev_recall,
        per_domain_jga,
        per_slot_acc,
        dev_error,
    ) = compute_jacc(
        data=all_dev_result,
        domains=domains,
        ignore_dontcare_in_pred=True,
        default_cleaning_flag=default_cleaning_flag,
    )
    dev_score = dst_joint_acc

    metrics_dict = {
        "Domain_Prediction_Accuracy": [0],
        "DST_Joint_Accuracy": [dst_joint_acc],
        "DST_Slot_F1": [dev_f1],
        "DST_Precision": [dev_precision],
        "DST_Recall": [dev_recall],
    }

    print(
        "Test Joint Accuracy is {}. Slot F1 is {}, Precision is {}, Recall is {}".format(
            dst_joint_acc, dev_f1, dev_precision, dev_recall
        )
    )
    print(
        "Number of total turns is {}, Number of valid turns is {}".format(total, valid)
    )

    pda = []
    print("\nPer Domain Accuracy:")
    for domain in per_domain_jga:
        print(
            "Domain: {}, Test Joint Accuracy is {}, F1 is {}, Total turns {}".format(
                domain,
                per_domain_jga[domain][0],
                per_domain_jga[domain][2],
                per_domain_jga[domain][1],
            )
        )
        pda.append(per_domain_jga[domain][0])
    print(f"AVG per domain JGA: {sum(pda) / len(pda)}")

    print("\nPer Slot Accuracy:")
    for domain in per_slot_acc:
        for slot in per_slot_acc[domain]:
            print(
                "{}-{}, Test Joint Accuracy is {}, Total turns {}".format(
                    domain,
                    slot,
                    per_slot_acc[domain][slot][0],
                    per_slot_acc[domain][slot][1],
                )
            )
    with open(dst_eval_error_path, "w") as file:
        json.dump(dev_error, file, indent=4)

    for domain in per_domain_jga:
        metrics_dict["DST_Joint_Accuracy_per_" + domain] = [per_domain_jga[domain][0]]
        metrics_dict["DST_Slot_F1_per_" + domain] = [per_domain_jga[domain][2]]

    metrics_df = pd.DataFrame(metrics_dict)

    metrics_csv_path = f"{data_prefix}/{split}{n_eval}-multi{multi_domain}-ref{ref_domain}-prev{add_prev}-{function_type}-dst{dst_nshot}shot-{model.replace('/', '#')}-metrics.csv"

    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Evaluation metrics saved successfully to {metrics_csv_path}")
    metrics_to_log = pprint_metrics(metrics_df)

    print(f"Task: {task} {type(task)}")
    # task = Task.current_task()
    # print(f"Task: {task} {type(task)}")

    if isinstance(task, Task):
        task.upload_artifact(
                "metrics",
                metrics_csv_path,
            )
        logger = task.get_logger()
        for metric in metrics_to_log.columns:
            value = metrics_to_log[metric].iloc[0]
            if isinstance(value, pd.Series):
                value = np.mean(value)
            value = float(value)
            logger.report_scalar(title="eval", series=metric, value=value, iteration=0)
    # sys.exit()
    # task.close()

    meta = {
        "datetime": str(datetime.now()),
        "training_method": "",
        "base_model": model,
        # "data_path": data_path,
        # "template": template,
        # "pd_size": pd_size,
        # "dataset_seed": dataset_seed,
        "queue": queue,
        # eval part
        "eval_dataset": dataset,
        "eval_split": split,
        "n_eval": n_eval,
        "ref_domain": ref_domain,
        "ref_bs": ref_bs,
        "multi_domain": multi_domain,
        "add_prev": add_prev,
        "function_type": function_type,
        "dst_nshot": dst_nshot,
        "nlg_nshot": nlg_nshot,
        "eval_template": template,
        "eval_model": model,
        "clearml_id": clearml_id,
        "eval_name": name,
        "handler": handler,
        "chat_template": chat_template,
        "eval_queue": queue,
        "temperature": temperature,
        "top_p": top_p,
        "generate": generate,
        "debug": debug,
        "verbose": verbose,
        "eval_valid": eval_valid,
        "uuid": save_path.split("/")[-1] if save_path else save_path,
        "reduce_inference_mem":reduce_inference_mem
    }
    if save_path:
        with open(os.path.join(save_path, "meta.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        metrics_df.to_csv(os.path.join(save_path, "metrics.csv"), index=False)
        print(f"{__name__}: Meta and metrics saved successfully to {save_path}.")

    return metrics_df


if __name__ == "__main__":
    fire.Fire(evaluate)
