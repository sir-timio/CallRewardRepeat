import gc
from time import sleep

import numpy as np
import torch
from datasets import Dataset


def stratified_by_functions(
    dataset: Dataset,
    functions_key: str,
    dialogs_per_function: int,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    function_to_indices = {}

    for idx, sample in enumerate(dataset):
        for function in sample[functions_key]:
            function_to_indices.setdefault(function["name"], []).append(idx)

    selected_functions = list(function_to_indices.keys())
    rng.shuffle(selected_functions)

    sampled_indices = set()
    for function in selected_functions:
        indices = function_to_indices[function]
        rng.shuffle(indices)
        sampled_indices.update(indices[:dialogs_per_function])

    return dataset.select(list(sampled_indices))


def cleanup_model():
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
        gc.collect()
    gc.collect()
    sleep(5)
