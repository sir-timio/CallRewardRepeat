import os
import json
import random


def unzip_session_data(data):
    dialog_turns = []
    for dial_id, turns in data.items():
        dialog_turns.extend(turns)
    return dialog_turns


def sample_data_ids(data, n):
    all_ids = list(data.keys())
    n = min(len(all_ids), n)
    sampled_ids = random.sample(all_ids, n)



def get_data_split(n_train=-1, n_val=-1, n_test=-1, return_list=False):
    processed_data_path = f"./data/pre-training_corpora/separate_datasets/Schema_Guided"

    with open(f"{processed_data_path}/schema_guided_train.json", "r") as file:
        train_data = json.load(file)
    with open(f"{processed_data_path}/schema_guided_val.json", "r") as file:
        dev_data = json.load(file)
    with open(f"{processed_data_path}/schema_guided_test.json", "r") as file:
        test_data = json.load(file)

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
