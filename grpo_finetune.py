import os
import shutil
import sys
from functools import partial

import fire
import orjson
import torch
from clearml import Dataset, InputModel, OutputModel, Task

from clearml_utils import create_task, execute_remotely, update_data_path, copy_data

if torch.cuda.is_available():
    os.system(f"pip install pydantic==2.10.6")
    # os.system("pip install flash_attn==2.8.3 --no-build-isolation")
    os.system(f"{sys.executable} -m pip install pydantic==2.10.6")
    # os.system(f"{sys.executable} -m pip install vllm==0.8.5")
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer
    from accelerate.utils import release_memory
    from chatbots.conversation import Conversation
    from chatbots.handlers.trained_model_handler import TrainedModelHandler

import datasets
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

from chatbots.handlers.utils import extract_first_dict
from create_grpo_finetunedata import main as sft2grpo_data
from utils import cleanup_model

seed = 421337
torch.seed = seed


def filter_by_prompt_length(example, tokenizer, cutoff_len):
    tokenized = tokenizer(
        example["prompt"],
        truncation=False,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return len(tokenized["input_ids"]) <= cutoff_len


class SavePeftModelCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_folder = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        kwargs["model"].save_pretrained(checkpoint_folder)
        return control


def train(
    base_model: str = "",
    data_path: str = "",
    output_dir: str = "",
    template: str = "",
    reward_func: str = "trained",
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
    train_on_response: bool = False,
    add_eos_token: bool = False,
    group_by_length: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",
    wandb_log_model: str = "",
    resume_from_checkpoint: str = None,
    debug: bool = False,
    num_generations: int = 4,
    use_vllm: bool = True,
    vllm_gpu_memory_utilization: float = 0.5,
    with_unsloth: bool = False,
    adapter_clearml_id: str = "",
    queue: str = "",
    save_merged: bool = False,
    dataset_seed: int = 42,
    pd_size: int = 50,
    task: Task = None,
    train_samples_limit: int = -1,
    n_slices: int = 1,
):
    print(bool(adapter_clearml_id))
    print(num_generations)
    if not base_model:
        raise ValueError("Please specify a --base_model")

    if task is not None:
        copy_data()

    else:
        task_name = f'{data_path.split("/")[-1].replace(".json", "")}_{base_model}_pd_{pd_size}_seed_{dataset_seed}_slices_{n_slices}'
        task = create_task(task_name)

    if queue:
        execute_remotely(task)

    update_data_path(data_path)

    gradient_accumulation_steps = batch_size // micro_batch_size

    if with_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            base_model,
            max_seq_length=cutoff_len + 1,
            fast_inference=True if use_vllm else False,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias = "none",
            random_state = 42
)


    else:
        if adapter_clearml_id != "null":
            full_model = InputModel(adapter_clearml_id).get_local_copy()
            model = AutoModelForCausalLM.from_pretrained(full_model)
            tokenizer = AutoTokenizer.from_pretrained(full_model)
        else:
            model = AutoModelForCausalLM.from_pretrained(base_model)
            tokenizer = AutoTokenizer.from_pretrained(base_model)

            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

    train_dataset = sft2grpo_data(
        input_file=data_path, save=False, limit=pd_size, seed=dataset_seed, n_slices=n_slices,
    )
    train_dataset = datasets.Dataset.from_list(train_dataset)


    print(f"dataset size: {len(train_dataset)}")


    train_dataset = train_dataset.filter(
        lambda example: filter_by_prompt_length(example, tokenizer, cutoff_len)
    )

    unique_dial_ids = list(set(train_dataset["dial_id"]))

    train_ids, test_ids = train_test_split(
        unique_dial_ids, test_size=0.1, random_state=seed
    )

    def filter_by_dial_ids(example, dial_ids_set):
        return example["dial_id"] in dial_ids_set

    train_ids_set = set(train_ids)
    test_ids_set = set(test_ids)

    eval_dataset = train_dataset.filter(lambda x: filter_by_dial_ids(x, test_ids_set))
    train_dataset = train_dataset.filter(lambda x: filter_by_dial_ids(x, train_ids_set))

    total_size = len(train_dataset)

    if train_samples_limit > 0:

        train_dataset = train_dataset.shuffle(seed=seed).select(
            range(min(len(train_dataset), train_samples_limit))
        )
        # eval_dataset = eval_dataset.shuffle(seed=seed).select(
        #     range(min(len(eval_dataset), 1000))
        # )

    print(f"filtered dataset size with taken rnd subsamples: {total_size}")

    def reward_function_partial(
        conv, handler: TrainedModelHandler, prompts, completions, debug=False, **kwargs
    ):
        rewards = []

        def _has_function(response: dict):
            return bool(response.get("function_call"))

        for i, (completion, answer) in enumerate(zip(completions, kwargs["answer"])):
            gt_parsed = handler.parse_response(answer, required=["function_call"])
            pred_parsed = handler.parse_response(completion, required=["function_call"])

            if not _has_function(gt_parsed) and not _has_function(pred_parsed):
                if debug:
                    print("#" * 50)
                    print(f"[{i}] ✅ No function in both → reward = 1")
                    print(f"completion: {completion}")
                    print(f"answer    : {answer}")
                    print("#" * 50)

                rewards.append(1.0)
                continue

            # === CASE 2: answer - no fc, completion - has fc ===
            if not _has_function(gt_parsed) and _has_function(pred_parsed):
                if debug:
                    print("#" * 50)
                    print(f"[{i}] ❌ Unexpected function in completion → reward = -1")
                    print(f"completion: {completion}")
                    print(f"answer    : {answer}")
                    print("#" * 50)
                rewards.append(-1.0)
                continue

            # === CASE 3: answer - has fc, completion - no fc ===
            if _has_function(gt_parsed) and not _has_function(pred_parsed):
                if debug:
                    print("#" * 50)
                    print(f"[{i}] ❌ Missing function in completion → reward = -1")
                    print(f"completion: {completion}")
                    print(f"answer    : {answer}")
                    print("#" * 50)

                rewards.append(-1.0)
                continue
            gt = gt_parsed["function_call"]
            pred = pred_parsed["function_call"]
            if gt.get("name") != pred.get("name"):
                rewards.append(0.0)
                continue
            gt_args, pred_args = gt.get("arguments", {}), pred.get("arguments", {})
            total = len(gt_args)
            if total == 0:  # same name, no-arg call
                rewards.append(1.0)
                continue
            try:
                hits = sum(
                    1 for k, v in gt_args.items() if k in pred_args and pred_args[k] == v
                )
                reward = hits / total  # fractional credit
            except:
                reward = -1
            rewards.append(reward)

            if debug:
                print(f"[{i}] ✅ Parsed successfully")
                print("→ GT:", gt)
                print("→ Pred:", pred)
                print(f"→ Reward: {rewards[-1]}")
                print(f"completion: {completion}")
                print(f"answer    : {answer}")

            if debug:
                _ = input()
        return rewards

    def reward_function_trained(
        conv, handler: TrainedModelHandler, prompts, completions, debug=False, **kwargs
    ):
        rewards = []

        def _has_function(response: dict):
            return bool(response.get("function_call"))

        for i, (completion, answer) in enumerate(zip(completions, kwargs["answer"])):
            gt_parsed = handler.parse_response(answer, required=["function_call"])
            pred_parsed = handler.parse_response(completion, required=["function_call"])

            if not _has_function(gt_parsed) and not _has_function(pred_parsed):
                if debug:
                    print("#" * 50)
                    print(f"[{i}] ✅ No function in both → reward = 1")
                    print(f"completion: {completion}")
                    print(f"answer    : {answer}")
                    print("#" * 50)

                rewards.append(1.0)
                continue

            # === CASE 2: answer - no fc, completion - has fc ===
            if not _has_function(gt_parsed) and _has_function(pred_parsed):
                if debug:
                    print("#" * 50)
                    print(f"[{i}] ❌ Unexpected function in completion → reward = -1")
                    print(f"completion: {completion}")
                    print(f"answer    : {answer}")
                    print("#" * 50)
                rewards.append(-1.0)
                continue

            # === CASE 3: answer - has fc, completion - no fc ===
            if _has_function(gt_parsed) and not _has_function(pred_parsed):
                if debug:
                    print("#" * 50)
                    print(f"[{i}] ❌ Missing function in completion → reward = -1")
                    print(f"completion: {completion}")
                    print(f"answer    : {answer}")
                    print("#" * 50)

                rewards.append(-1.0)
                continue
            gt = gt_parsed["function_call"]
            pred = pred_parsed["function_call"]
            match = (
                isinstance(gt, dict)
                and isinstance(pred, dict)
                and gt.get("name") == pred.get("name")
                and gt.get("arguments") == pred.get("arguments")
            )

            reward = 1.0 if match else 0.0
            if debug:
                print(f"[{i}] ✅ Parsed successfully")
                print("→ GT:", gt)
                print("→ Pred:", pred)
                print("→ Match:", match)
                print(f"→ Reward: {reward}")
                print(f"completion: {completion}")
                print(f"answer    : {answer}")

            rewards.append(reward)
            if debug:
                _ = input()
        return rewards

    def reward_function_untrained(
        fc_prefix, fc_suffix, prompts, completions, debug=False, **kwargs
    ):
        rewards = []

        for i, (completion, answer) in enumerate(zip(completions, kwargs["answer"])):

            def _has_function(text):
                return fc_prefix in text and fc_suffix in text

            # === CASE 1: no fc (in both) ===
            if not _has_function(answer) and not _has_function(completion):
                if debug:
                    print("#" * 50)
                    print(f"[{i}] ✅ No function in both → reward = 1")
                    print(f"completion: {completion}")
                    print(f"answer    : {answer}")
                    print("#" * 50)

                rewards.append(1.0)
                continue

            # === CASE 2: answer - no fc, completion - has fc ===
            if not _has_function(answer) and _has_function(completion):
                if debug:
                    print("#" * 50)
                    print(f"[{i}] ❌ Unexpected function in completion → reward = -1")
                    print(f"completion: {completion}")
                    print(f"answer    : {answer}")
                    print("#" * 50)
                rewards.append(-1.0)
                continue

            # === CASE 3: answer - has fc, completion - no fc ===
            if _has_function(answer) and not _has_function(completion):
                if debug:
                    print("#" * 50)
                    print(f"[{i}] ❌ Missing function in completion → reward = -1")
                    print(f"completion: {completion}")
                    print(f"answer    : {answer}")
                    print("#" * 50)

                rewards.append(-1.0)
                continue

            # === CASE 4: both have fc ===
            try:
                # extract inner JSONs
                answer = answer.lower().replace("\_", "_").replace("\n", " ").strip()
                completion = (
                    completion.lower().replace("\_", "_").replace("\n", " ").strip()
                )

                # answer_json_str = answer.split(fc_prefix)[1].split(fc_suffix)[0].strip()
                # completion_json_str = completion.split(fc_prefix)[1].split(fc_suffix)[0].strip()

                answer_json_str = answer.replace("```", "").replace("'", '"')
                completion_json_str = completion.replace("```", "").replace("'", '"')

                answer_json_str = extract_first_dict(answer_json_str)
                completion_json_str = extract_first_dict(completion_json_str)

                # answer_json_str = answer_json_str.replace("```", "").replace("'", '"')
                # completion_json_str = completion_json_str.replace("```", "").replace("'", '"')

                gt = orjson.loads(answer_json_str)
                pred = orjson.loads(completion_json_str)

                if isinstance(pred, list):
                    pred = pred[0]
                if not isinstance(pred, dict):
                    reward = 0
                    continue
                # strict name and args matching
                match = (
                    isinstance(gt, dict)
                    and isinstance(pred, dict)
                    and gt.get("name") == pred.get("name")
                    and gt.get("arguments") == pred.get("arguments")
                )

                reward = 1.0 if match else 0.0
                if debug:
                    print(f"[{i}] ✅ Parsed successfully")
                    print("→ GT:", gt)
                    print("→ Pred:", pred)
                    print("→ Match:", match)
                    print(f"→ Reward: {reward}")
                    print(f"completion: {completion}")
                    print(f"answer    : {answer}")

                rewards.append(reward)
                # if debug:
                #     _ = input()

            except Exception as e:
                if debug:
                    print(f"[{i}] ❌ Error during parsing → reward = -1")
                    print("Error:", e)
                rewards.append(-1.0)

        return rewards

    if reward_func == "trained":
        conv = Conversation(template)
        handler = TrainedModelHandler(
            model_name=base_model,
            template=template,
            gpu_memory_utilization=0,
            skip_llm_init=True,
        )

        reward_function = partial(
            reward_function_trained, conv=conv, handler=handler, debug=debug
        )
        reward_function.__name__ = "FC"
    elif reward_func == "partial":
        conv = Conversation(template)
        handler = TrainedModelHandler(
            model_name=base_model,
            template=template,
            gpu_memory_utilization=0,
            skip_llm_init=True,
        )

        reward_function = partial(
            reward_function_partial, conv=conv, handler=handler, debug=debug
        )
        reward_function.__name__ = "FC"
    else:
        reward_function = partial(
            reward_function_untrained, debug=debug, fc_prefix="{", fc_suffix="}"
        )
        reward_function.__name__ = "FC"

    save_steps = 500
    eval_steps = 2000
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        save_strategy="epoch",
        save_steps=save_steps,
        #eval_strategy="epoch",
        eval_strategy="no",
        #eval_steps=eval_steps,
        max_grad_norm=0.5,
        warmup_steps=warmup_steps,
        lr_scheduler_type=lr_scheduler,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        logging_steps=10,
        beta=0.004,
        bf16=True,
        report_to=["clearml"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=use_vllm,
        max_prompt_length=cutoff_len,
        max_completion_length=128,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        num_generations=num_generations,
        save_total_limit=1,
    )
    # def apply_chat_template(sample):
    #     query = sample.get("query", "")
    #     tools = json.loads(sample.get("tools", "[]"))
    #     answers = sample.get("answers", "[]")
    #     prompt = [{"role": "user", "content": query}]
    #     completion = [{"role": "assistant", "content": answers}]
    #     if hasattr(tokenizer, "apply_chat_template"):
    #         prompt = tokenizer.apply_chat_template(prompt, tools, tokenize=False)
    #         completion = tokenizer.apply_chat_template(completion, tokenize=False)
    #     return {"prompt": prompt, "completion": completion}

    # train_dataset = dataset.map(lambda x: apply_chat_template(x))
    # print("USING REWARD FOR UNTRAINED")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_function,
        eval_dataset=eval_dataset,
        train_dataset=train_dataset,
        args=grpo_config,
        callbacks=[SavePeftModelCallback()],
    )
    trainer.train()

    outmodel = OutputModel()
    #ray.shutdown()
    #del trainer.llm
    # if hasattr(trainer, "llm"):
    #     print(f"Removing llm client: {trainer.llm}")
    #     del trainer.llm
    # if hasattr(trainer, "vllm_client"):
    #     print(f"Removing vllm_client: {trainer.vllm_client}")
    #     del trainer.vllm_client
    # trainer, model = release_memory(trainer, model)
    del trainer, model
    cleanup_model()

    # outmodel.update_weights(output_dir, auto_delete_file=False)

    from merge import load_peft_with_fallback
    device_map = "auto"
    init_model = AutoModelForCausalLM.from_pretrained(base_model, device_map=device_map,
                                                      torch_dtype=torch.bfloat16)
    peft_model = load_peft_with_fallback(init_model, output_dir)
    peft_model = peft_model.merge_and_unload()
    peft_model.save_pretrained("merged")
    tokenizer.save_pretrained("merged")

    init_model, peft_model = release_memory(init_model, peft_model)
    del init_model, peft_model
    cleanup_model()
    cleanup_model()

    if save_merged:
        outmodel.update_weights("merged", auto_delete_file=False)
        cleanup_model()
    else:  # save only lora
        outmodel.update_weights(output_dir, auto_delete_file=False)
        cleanup_model()
    return task if queue else None
    # outmodel.update_weights(output_dir, auto_delete_file=False)

    # model = model.merge_and_unload()
    # model.save_pretrained("merged")
    # tokenizer.save_pretrained("merged")
    # outmodel = OutputModel()
    # outmodel.update_weights("merged", auto_delete_file=False)
    # task.close()


if __name__ == "__main__":
    fire.Fire(train)
