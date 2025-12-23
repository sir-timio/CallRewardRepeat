import os
import sys
from typing import List

import fire
import torch
from clearml import Dataset, OutputModel, Task

from clearml_utils import create_task, execute_remotely, update_data_path

if torch.cuda.is_available():
    # os.system("pip install flash_attn==2.8.3 --no-build-isolation")
    os.system(f"{sys.executable} -m spacy download en_core_web_sm")
    from unsloth import FastLanguageModel
    from accelerate.utils import release_memory

from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq,  TrainerCallback, set_seed)
from trl import SFTConfig, SFTTrainer

from chatbots.conversation import Conversation
from utils import cleanup_model, stratified_by_functions
from merge import load_peft_with_fallback

set_seed(42)

def count_trainable_tokens(dataset):
    trainable_count = 0
    total_count = 0
    for item in dataset:
        trainable_count += sum(1 for label in item["labels"] if label != -100)
        total_count += len(item["labels"])
    print(f"Trainable tokens: {trainable_count}/{total_count}")


class SavePeftModelCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_folder = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        kwargs["model"].save_pretrained(checkpoint_folder)
        return control


class LoadBestPeftModelCallback(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        print(f"Loading best PEFT model from {state.best_model_checkpoint}")
        best_model_path = os.path.join(state.best_model_checkpoint, "adapter_model.bin")
        adapters_weights = torch.load(best_model_path)
        model = kwargs["model"]
        set_peft_model_state_dict(model, adapters_weights)
        return control


def train(
    # model/data params
    base_model: str = "",
    data_path: str = "",
    output_dir: str = "",
    template: str = "",
    # training hyperparam
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
    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # from peft docs: ["q_proj", "k_proj", "v_proj", "o_proj", "fc_in", "fc_out", "wte", "gate_proj", "down_proj", "up_proj"]
    lora_target_modules: List[str] = ["gate_proj", "down_proj", "up_proj"],
    with_unsloth: bool = False,
    # llm hyperparams
    train_on_response: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    queue: str = "",
    save_merged: bool = False,
    dataset_seed: int = 42,
    pd_size: int = 50,
    training_type: str = "sft",
    task:Task = None,
    use_fp8:bool = False,
):
    print(f"base_model: {base_model}")
    print(f"data path: {data_path}")
    print(f"dataset pd_size: {pd_size}")
    print(f"dataset seed: {dataset_seed}")
    print(f"FP8: {use_fp8}")

    if not task:
        task_name = f'{training_type.upper()}_{data_path.split("/")[-1].replace(".json", "")}_{base_model}_pd_{pd_size}_seed_{dataset_seed}'
        task = create_task(task_name)
        if queue:
            execute_remotely(task)

    data_path = update_data_path(data_path)

    if not base_model:
        raise ValueError("Please specify a --base_model")
    conv = Conversation(template_name=template)

    fc_prefix = conv.function_call_prefix
    fc_suffix = conv.function_call_suffix

    gradient_accumulation_steps = batch_size // micro_batch_size
    device_map = "auto"

    print(f"Training with base model: {base_model}")
    print(
        f"Template: {template}, function call suffix and prefix: {(fc_prefix, fc_suffix)}"
    )

    if with_unsloth:
        if use_fp8:
            print("Using FP8")
        model, tokenizer = FastLanguageModel.from_pretrained(
            base_model,
            max_seq_length=cutoff_len + 1,
            load_in_4bit=False,
            # load_in_8bit=False,
            # load_in_16bit=not use_fp8,
            # dtype=torch.bfloat16,
            # load_in_fp8 = use_fp8,
            # fast_inference = use_fp8,
            use_gradient_checkpointing = "unsloth"
        )
        print(model.dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model, device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained(base_model)

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_conversation(data_point):
        # data in the conversation fashion
        system_prompt = data_point["system"]
        conversation = data_point["conversation"]

        labels = []
        tokenized_ids = []
        target_len = 0

        # system message part
        tokenized_system_prompt = tokenize(system_prompt, add_eos_token=add_eos_token)
        system_prompt_len = len(tokenized_system_prompt["input_ids"])
        if add_eos_token:
            system_prompt_len -= 1
        tokenized_ids += tokenized_system_prompt["input_ids"]
        labels += [-100] * system_prompt_len

        # conversation
        for message in conversation:
            ###### user turn ######
            if message["role"] == "user":
                user_input = message["content"]
                tokenized_user_input = tokenize(user_input, add_eos_token=add_eos_token)
                user_input_len = len(tokenized_user_input["input_ids"])
                if add_eos_token:
                    user_input_len -= 1
                tokenized_ids += tokenized_user_input["input_ids"]
                labels += [-100] * user_input_len

            ###### assistant turn ######
            elif message["role"] == "assistant":
                assistant_output = message["content"]

                if fc_prefix in assistant_output and fc_suffix in assistant_output:
                    prefix_start = assistant_output.find(fc_prefix)
                    suffix_start = assistant_output.rfind(fc_suffix)

                    if (
                        prefix_start != -1
                        and suffix_start != -1
                        and suffix_start > prefix_start
                    ):
                        suffix_end = suffix_start + len(fc_suffix)
                        function_call = assistant_output[prefix_start:suffix_end]
                        response = assistant_output[suffix_end:]
                    else:
                        function_call = ""
                        response = assistant_output
                else:
                    function_call = ""
                    response = assistant_output

                ### part 1: function call ###
                if function_call:
                    tokenized_function_call = tokenize(
                        function_call, add_eos_token=add_eos_token
                    )
                    function_call_output_len = len(tokenized_function_call["input_ids"])
                    if add_eos_token:
                        function_call_output_len -= 1
                    tokenized_ids += tokenized_function_call["input_ids"]
                    labels += tokenized_function_call["input_ids"]
                    target_len += function_call_output_len

                ### part 2: response ###
                if response:
                    tokenized_response = tokenize(response, add_eos_token=add_eos_token)
                    response_output_len = len(tokenized_response["input_ids"])
                    if add_eos_token:
                        response_output_len -= 1
                    tokenized_ids += tokenized_response["input_ids"]
                    if train_on_response:
                        labels += tokenized_response["input_ids"]
                        target_len += response_output_len
                    else:
                        labels += [-100] * response_output_len

        assert len(tokenized_ids) == len(labels)

        return {
            "input_ids": torch.LongTensor(tokenized_ids),
            "labels": torch.LongTensor(labels),
        }

    # model = prepare_model_for_int8_training(model)

    if with_unsloth:
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

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            task_type="CAUSAL_LM",
            # use_rslora=True,
        )

        model = get_peft_model(model, config)

    data = (
        load_dataset("json", data_files=data_path)
        if data_path.endswith(".json")
        else load_dataset(data_path)
    )

    def analyze_function_calls(data):
        function_call_count = 0
        total_turns = 0
        for dialog in data:
            for message in dialog["conversation"]:
                if message["role"] == "assistant":
                    total_turns += 1
                    if (
                        fc_prefix in message["content"]
                        and fc_suffix in message["content"]
                    ):
                        function_call_count += 1

        print(
            f"Function calls: {function_call_count}/{total_turns} ({function_call_count/total_turns*100:.2f}%)"
        )
        return function_call_count > 0

    has_function_calls = analyze_function_calls(data["train"])
    if not has_function_calls:
        print("WARNING: No function calls found in the dataset!")
    if pd_size != -1:
        data["train"] = stratified_by_functions(
            data["train"],
            functions_key="functions",
            dialogs_per_function=pd_size,
            seed=dataset_seed,
        )

    # print(data_path)
    # print(1/0)
    # tokenized_data = data.map(tokenize_function, batched=True)
    tokenized_data = data.map(
        generate_and_tokenize_conversation,
        remove_columns=data["train"].column_names,
        batched=False,
    )
    train_dataset = tokenized_data["train"]
    train_dataset = train_dataset.filter(
        lambda x: any(label != -100 for label in x["labels"])
    )

    print(f"After sampling and filter: {len(train_dataset)}")

    count_trainable_tokens(train_dataset)
    if val_set_size > 0:
        train_val_split = train_dataset.train_test_split(test_size=val_set_size)
        train_dataset, val_dataset = train_val_split["train"], train_val_split["test"]
        eval_strategy = "steps"
    else:
        val_dataset = None
        eval_strategy = "no"

    # model.gradient_checkpointing_enable()

    trainer_args = SFTConfig(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        logging_dir="./logs",
        logging_steps=10,
        # evaluation_strategy=eval_strategy,
        eval_steps=eval_steps if val_set_size > 0 else None,
        save_strategy="epoch",
        save_steps=save_steps,
        output_dir=output_dir,
        save_total_limit=1,
        bf16=True,
        report_to="clearml",
        run_name="PEFT_FineTuning",
        # use_liger_kernel=True

    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if val_set_size > 0 else None,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, max_length=cutoff_len, padding=True
        ),
        callbacks=[SavePeftModelCallback()],
    )
    trainer.train()
    outmodel = OutputModel()
    trainer, model = release_memory(trainer, model)
    del trainer, model
    cleanup_model()

    # outmodel.update_weights(output_dir, auto_delete_file=False)


    init_model = AutoModelForCausalLM.from_pretrained(base_model, device_map=device_map, torch_dtype=torch.bfloat16)
    peft_model = load_peft_with_fallback(init_model, output_dir)
    peft_model = peft_model.merge_and_unload()
    peft_model.save_pretrained("merged")
    tokenizer.save_pretrained("merged")

    init_model, peft_model = release_memory(init_model, peft_model)
    del init_model, peft_model
    cleanup_model()
    cleanup_model()

    if save_merged:
        print("Uploading merged model to clearml")
        outmodel.update_weights("merged", auto_delete_file=False)
        cleanup_model()
    else:  # save only lora
        print("Uploading lora model to clearml")
        outmodel.update_weights(output_dir, auto_delete_file=False)
        cleanup_model()
    return task if queue else None


if __name__ == "__main__":
    fire.Fire(train)
