import json
import os
from datetime import datetime
from uuid import uuid4

import fire

from inference import evaluate


def zero_shot_evaluate(
    save_dir: str = "DST_scores/zero-shot",
    base_model: str = "",
    queue: str = "",
    eval_datasets: str = "mwoz21",
    eval_split: str = "test",
    n_eval: int = 100,
    ref_domain: bool = True,
    ref_bs: bool = False,
    multi_domain: bool = False,
    add_prev: bool = True,
    function_type: str = "json",
    dst_nshot: int = 0,
    nlg_nshot: int = 0,
    eval_template: str = "",
    model: str = "",
    clearml_id: str = "null",
    eval_name: str = "",
    handler: str = "",
    chat_template: str = "",
    temperature: float = 0.3,
    top_p: float = 0.2,
    generate: bool = True,
    debug: bool = False,
    verbose: bool = False,
    eval_valid: bool = False,
    gpu_memory_utilization: float = 0.8,
    reduce_inference_mem:bool=False
):
    # 2. evaluate
    # model is "merged"
    for eval_dataset in eval_datasets.split(";"):
        save_path = f"{save_dir}/{str(uuid4())}"
        print(f"SAVING PATH: {save_path}")

        metrics_df = evaluate(
            queue=queue,
            dataset=eval_dataset,
            split=eval_split,
            n_eval=n_eval,
            ref_domain=ref_domain,
            ref_bs=ref_bs,
            multi_domain=multi_domain,
            add_prev=add_prev,
            function_type=function_type,
            dst_nshot=dst_nshot,
            nlg_nshot=nlg_nshot,
            template=eval_template,
            model=base_model,
            clearml_id=clearml_id,
            name=eval_name,
            handler=handler,
            chat_template=chat_template,
            temperature=temperature,
            top_p=top_p,
            generate=generate,
            debug=debug,
            verbose=verbose,
            eval_valid=eval_valid,
            task=None,
            gpu_memory_utilization=gpu_memory_utilization,
            reduce_inference_mem=reduce_inference_mem
            # save_dir=save_dir
        )
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_path, exist_ok=True)

        meta = {
            "datetime": str(datetime.now()),
            "training_method": "zero-shot",
            "base_model": base_model,
            "pd_size": 0,
            "dataset_seed": 42,
            "train_dataset": "No",
            # eval part
            "eval_dataset": eval_dataset,
            "eval_split": eval_split,
            "n_eval": n_eval,
            "ref_domain": ref_domain,
            "ref_bs": ref_bs,
            "multi_domain": multi_domain,
            "add_prev": add_prev,
            "function_type": function_type,
            "dst_nshot": dst_nshot,
            "nlg_nshot": nlg_nshot,
            "eval_template": eval_template,
            "clearml_id": clearml_id,
            "eval_name": eval_name,
            "handler": handler,
            "chat_template": chat_template,
            "temperature": temperature,
            "top_p": top_p,
            "generate": generate,
            "debug": debug,
            "verbose": verbose,
            "eval_valid": eval_valid,
            "uuid": save_path.split("/")[-1],
            "reduce_inference_mem":reduce_inference_mem
        }
        with open(os.path.join(save_path, "meta.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        metrics_df.to_csv(os.path.join(save_path, "metrics.csv"), index=False)
        print(f"Meta and metrics saved successfully to {save_path}.")


if __name__ == "__main__":
    fire.Fire(zero_shot_evaluate)
