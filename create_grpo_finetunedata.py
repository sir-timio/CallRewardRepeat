import argparse
import json
import random
from uuid import uuid4

import numpy as np
from datasets import Dataset
from scipy.special import factorial
from tqdm import tqdm


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True, help="Path to input JSON file with preprocessed dialogues")
    parser.add_argument("--output-file", type=str, required=True, help="Path to output JSONL file with prompt-completion slices")
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of dialogues processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=argparse.BooleanOptionalAction, default=True, help="Save the output to a file")
    return parser.parse_args()


def skewed_choice(items, n, skew_factor=2.0):
    if not items:
        raise ValueError("List of items must not be empty")

    # Create weights, increasing by the end of the list
    indices = np.arange(1, len(items) + 1)
    probabilities = indices ** skew_factor
    probabilities /= probabilities.sum()

    return np.random.choice(items, size=n, replace=False, p=probabilities)

def poisson_skewed_choices(items, n=1, lam=2.0):
    """
    Makes a random choice from the list items with a Poisson distribution,
    skewed to the right end of the list

    Args:
        items (list): List of elements to choose from
        lam (float): λ (lambda) for the Poisson distribution

    Returns:
        Randomly selected items elements
    """
    if not items:
        raise ValueError("List of items must not be empty")

    size = len(items)
    indices = np.arange(size)

    # Generate Poisson distribution probabilities, skewed to the right
    probabilities = np.exp(-lam) * lam ** indices / factorial(indices)
    probabilities = probabilities[::-1]  # Reverse to skew to the right
    probabilities /= probabilities.sum()

    return np.random.choice(items, p=probabilities, size=n)

def slice_dialog(dialogue):
    system_prompt = dialogue.get("system", "")
    messages = dialogue["conversation"]
    dial_id = dialogue.get("dial_id") or str(uuid4())

    context = [{"role": "system", "content": system_prompt}]
    slices = []

    for message in messages:
        if message["role"] == "user":
            context.append(message)
        elif message["role"] == "assistant":
            prompt = "".join([m["content"] for m in context[:]])
            completion = message["content"]

            slices.append({
                "prompt": prompt.strip(),
                "answer": completion.strip(),
                "dial_id": dial_id,
            })

            context.append(message)

    return slices


def stratified_by_functions(
    dataset: list,
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

    return [dataset[i] for i in sampled_indices]


def main(input_file, output_file=None, save=True, limit=1, seed=42, n_slices=1):
    if save:
        assert output_file is not None, "Output file must be specified when saving"
    random.seed(seed)
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # random.shuffle(data)  # Shuffle the data to ensure randomness in slicing
    # if limit > 0:
    #     data = data[:limit]

    if limit > 0:
        data = stratified_by_functions(
                data,
                functions_key="functions",
                dialogs_per_function=limit,
                seed=seed,
            )

    all_slices = []
    for dial in tqdm(data, desc="Processing dialogues"):
        slices = slice_dialog(dial)
        if n_slices > 0:
            chosen_slices = poisson_skewed_choices([s for s in slices if "{" in s['answer']], n=n_slices)
        else:
            chosen_slices = slices
        random.shuffle(chosen_slices)
        all_slices.extend(chosen_slices)

    if save:
        with open(output_file, "w", encoding="utf-8") as f:
            for ex in all_slices:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"Saved {len(all_slices)} prompt→completion pairs to {output_file}")
    return all_slices


if __name__ == "__main__":
    args = read_args()
    main(**vars(args))
