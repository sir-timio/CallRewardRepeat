import argparse
import os
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_peft_with_fallback(base_model, peft_path: str | Path, **device_arg):
    peft_path = Path(peft_path)

    try:
        print(f"Loading PEFT: {peft_path}")
        return PeftModel.from_pretrained(base_model, peft_path.as_posix(), **device_arg)
    except Exception as e:
        print(f"⚠️  Can't load PEFT from {peft_path}: {e}")

        checkpoints: list[tuple[int, Path]] = []
        for sub in peft_path.iterdir():
            if sub.is_dir():
                m = re.match(r"checkpoint-(\d+)$", sub.name)
                if m:
                    checkpoints.append((int(m.group(1)), sub))

        if not checkpoints:
            raise RuntimeError(
                f"No checkpoint dir in  {peft_path}; nothing to load"
            ) from e

        step, best_ckpt = max(checkpoints, key=lambda t: t[0])
        print(f"Trying to load checkpoint‑{step} again: {best_ckpt}")

        return PeftModel.from_pretrained(base_model, best_ckpt.as_posix(), **device_arg)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str)
    parser.add_argument("--peft_model_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


def main():
    args = get_args()

    if args.device == "auto":
        device_arg = {"device_map": "auto"}
    else:
        device_arg = {"device_map": {"": args.device}}

    print(f"Loading base model: {args.base_model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float16,
        **device_arg,
    )

    print(f"Loading PEFT: {args.peft_model_path}")
    # model = PeftModel.from_pretrained(base_model, args.peft_model_path, **device_arg)
    model = load_peft_with_fallback(base_model, args.peft_model_path, **device_arg)
    print(f"Running merge_and_unload")
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

    model.save_pretrained(f"{args.output_dir}")
    tokenizer.save_pretrained(f"{args.output_dir}")
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
