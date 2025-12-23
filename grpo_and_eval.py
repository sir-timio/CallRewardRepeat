import json
import os
from datetime import datetime
from uuid import uuid4

import fire
from clearml import Task

from grpo_finetune import train
from inference import evaluate


def train_and_evaluate(
    # general
    save_dir: str = "DST_scores/GRPO",
    base_model: str = "",
    data_path: str = "",
    output_dir: str = "",
    # train
    template: str = "",
    pd_size: int = 50,
    dataset_seed: int = 50,
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    cutoff_len: int = 4096,
    val_set_size: int = 0,
    eval_steps: int = 200,
    save_steps: int = 1000,
    lr_scheduler: str = "cosine",
    warmup_steps: int = 100,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules=["gate_proj", "down_proj", "up_proj"],
    with_unsloth: bool = False,
    train_on_response: bool = False,
    add_eos_token: bool = False,
    group_by_length: bool = False,
    resume_from_checkpoint: str = None,
    adapter_clearml_id: str = "",
    queue: str = "",
    save_merged: bool = False,
    train_dataset: str = "mwoz",
    reward_func: str = "trained",
    num_generations: int = 4,
    use_vllm_for_grpo: bool = True,
    grpo_gpu_memory_utilization: float = 0.5,
    train_samples_limit: int = -1,
    n_slices: int = 1,
    # eval
    eval_datasets: str = "mwoz",
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
    clearml_id: str = "null",
    eval_name: str = "",
    handler: str = "",
    chat_template: str = "",
    eval_queue: str = "",
    temperature: float = 0.3,
    top_p: float = 0.2,
    generate: bool = True,
    debug: bool = False,
    verbose: bool = False,
    eval_valid: bool = False,
    eval_gpu_memory_utilization: float = 0.8,
    reduce_inference_mem: bool = False
):
    # 1. train
    clearml_task = train(
        base_model=base_model,
        data_path=data_path,
        output_dir=output_dir,
        template=template,
        pd_size=pd_size,
        dataset_seed=dataset_seed,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        cutoff_len=cutoff_len,
        val_set_size=val_set_size,
        eval_steps=eval_steps,
        save_steps=save_steps,
        lr_scheduler=lr_scheduler,
        warmup_steps=warmup_steps,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        with_unsloth=with_unsloth,
        train_on_response=train_on_response,
        add_eos_token=add_eos_token,
        group_by_length=group_by_length,
        resume_from_checkpoint=resume_from_checkpoint,
        queue=queue,
        save_merged=save_merged,
        reward_func=reward_func,
        num_generations=num_generations,
        use_vllm=use_vllm_for_grpo,
        vllm_gpu_memory_utilization=grpo_gpu_memory_utilization,
        adapter_clearml_id=adapter_clearml_id,
        train_samples_limit=train_samples_limit,
        n_slices=n_slices,

    )
    # 2. evaluate
    eval_model = "merged"
    for eval_dataset in eval_datasets.split(";"):
        save_path = f"{save_dir}/{str(uuid4())}"
        os.makedirs(save_path, exist_ok=True)
        print(f"Evaluation on {eval_dataset}")
        print(f"SAVING PATH: {save_path}")
        metrics_df = evaluate(
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
            model=eval_model,
            clearml_id=clearml_id,
            name=eval_name,
            handler=handler,
            chat_template=chat_template,
            queue=eval_queue,
            temperature=temperature,
            top_p=top_p,
            generate=generate,
            debug=debug,
            verbose=verbose,
            eval_valid=eval_valid,
            task=clearml_task,
            gpu_memory_utilization=eval_gpu_memory_utilization,
            # save_dir=save_dir,
            reduce_inference_mem=reduce_inference_mem
        )

        meta = {
            "datetime": str(datetime.now()),
            "training_method": "GRPO",
            "base_model": base_model,
            "data_path": data_path,
            "output_dir": output_dir,
            "template": template,
            "pd_size": pd_size,
            "dataset_seed": dataset_seed,
            "train_dataset": train_dataset,
            "batch_size": batch_size,
            "micro_batch_size": micro_batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "cutoff_len": cutoff_len,
            "val_set_size": val_set_size,
            "eval_steps": eval_steps,
            "save_steps": save_steps,
            "lr_scheduler": lr_scheduler,
            "warmup_steps": warmup_steps,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_target_modules": lora_target_modules,
            "with_unsloth": with_unsloth,
            "train_on_response": train_on_response,
            "add_eos_token": add_eos_token,
            "group_by_length": group_by_length,
            "resume_from_checkpoint": resume_from_checkpoint,
            "queue": queue,
            "save_merged": save_merged,
            "reward_func": reward_func,
            "train_samples_limit": train_samples_limit,
            "n_slices": n_slices,
            "task_id":  clearml_task.id if isinstance(clearml_task, Task) else None,
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
            "eval_model": eval_model,
            "clearml_id": clearml_id,
            "eval_name": eval_name,
            "handler": handler,
            "chat_template": chat_template,
            "eval_queue": eval_queue,
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
    fire.Fire(train_and_evaluate)
