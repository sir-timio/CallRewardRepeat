import os
import json
import random


def unzip_session_data(data):
    dialog_turns = []
    for dial_id, turns in data.items():
        dialog_turns.extend(turns)
    return dialog_turns


def sample_data_ids(data, n):
    domain_freq = {
        "[taxi]": 0,
        "[hotel]": 0,
        "[train]": 0,
        "[restaurant]": 0,
        "[attraction]": 0,
        "[profile]": 0,
        "[taxi]": 0,
    }

    all_ids = list(data.keys())
    n = min(len(all_ids), n)
    sampled_ids = random.sample(all_ids, n)
    for sampled_id in sampled_ids:
        turns = data[sampled_id]
        for turn in turns:
            if turn["dspn"] in domain_freq:
                domain_freq[turn["dspn"]] += 1
    print(domain_freq)

    return sampled_ids


def get_data_split(n_train=-1, n_val=-1, n_test=-1, return_list=False):
    processed_data_path = f"./data/spokenwoz/"

    with open(f"{processed_data_path}/train/processed.json", "r") as file:
        train_dev_data = json.load(file)
    with open(f"{processed_data_path}/test/processed.json", "r") as file:
        test_data = json.load(file)
    val_list = []
    with open(f"{processed_data_path}/train/valListFile.json", "r") as file:
        for line in file:
            val_list.append(line.strip("\n"))
    val_list = set(val_list)
    train_data = {}
    val_data = {}
    for k, v in train_dev_data.items():
        if k in val_list:
            val_data[k] = v
        else:
            train_data[k] = v

    if n_train != -1:
        if not os.path.exists(f"{processed_data_path}/train_{n_train}_ids.json"):
            train_data_ids = sample_data_ids(train_data, n_train)
            if n_train < len(
                train_data
            ):  # only record the sampled ids is not the full set
                with open(
                    f"{processed_data_path}/train_{n_train}_ids.json", "w"
                ) as file:
                    json.dump(train_data_ids, file)
        else:
            with open(f"{processed_data_path}/train_{n_train}_ids.json", "r") as file:
                train_data_ids = json.load(file)

        sampled_train_data = {}
        for did in train_data_ids:
            sampled_train_data[did] = train_data[did]
        train_data = sampled_train_data

    if n_val != -1:
        if not os.path.exists(f"{processed_data_path}/dev_{n_val}_ids.json"):
            val_data_ids = sample_data_ids(val_data, n_val)
            if n_val < len(val_data):  # only record the sampled ids is not the full set
                with open(f"{processed_data_path}/dev_{n_val}_ids.json", "w") as file:
                    json.dump(val_data_ids, file)
        else:
            with open(f"{processed_data_path}/dev_{n_val}_ids.json", "r") as file:
                val_data_ids = json.load(file)

        sampled_val_data = {}
        for did in val_data_ids:
            sampled_val_data[did] = val_data[did]
        val_data = sampled_val_data

    if n_test != -1:
        if not os.path.exists(f"{processed_data_path}/test_{n_test}_ids.json"):
            test_data_ids = sample_data_ids(test_data, n_test)
            if n_test < len(
                test_data
            ):  # only record the sampled ids is not the full set
                with open(f"{processed_data_path}/test_{n_test}_ids.json", "w") as file:
                    json.dump(test_data_ids, file)
        else:
            with open(f"{processed_data_path}/test_{n_test}_ids.json", "r") as file:
                test_data_ids = json.load(file)

        sampled_test_data = {}
        for did in test_data_ids:
            sampled_test_data[did] = test_data[did]
        test_data = sampled_test_data

    if return_list:
        return (
            unzip_session_data(train_data),
            unzip_session_data(val_data),
            unzip_session_data(test_data),
        )
    else:
        return train_data, val_data, test_data
