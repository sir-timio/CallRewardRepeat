export TRANSFORMERS_CACHE='HOME_PATH/.cache/huggingface/transformers'
export HF_HOME='HOME_PATH/.cache/huggingface'

devices=0

batch_size=2
micro_batch_size=1
lora_r=16
lora_alpha=16

cd ..
# unsloth/Meta-Llama-3.1-8B-Instruct
# Team-ACE/ToolACE-2-Llama-3.1-8B
# Qwen/Qwen2.5-7B-Instruct
# for data in sft-llama2-200

for data in sft-qwen-pd100 sft-qwen-pd400

do
    for train_on_response in False
    do
        for lr in 0.0003
        do
            for ep in 1
            do
                CUDA_VISIBLE_DEVICES=$devices python finetune.py \
                    --base_model Qwen/Qwen2.5-0.5B-Instruct \
                    --data-path ./data/finetunedata/$data.json \
                    --output_dir ./ckpt/lora_ckpt/qwen25-0.5B-chat-$data \
                    --wandb_project toolace-$data \
                    --wandb_watch all \
                    --batch_size $batch_size \
                    --micro_batch_size $micro_batch_size \
                    --num_epochs $ep \
                    --learning_rate $lr \
                    --cutoff_len 4096 \
                    --val_set_size 0 \
                    --lora_r $lora_r \
                    --lora_alpha $lora_alpha \
                    --lora_dropout 0.05 \
                    --lora_target_modules '[q_proj, v_proj]' \
                    --train_on_response $train_on_response \
                    --add_eos_token False \
                    --group_by_length False \
                    --lr_scheduler 'cosine' \
                    --warmup_steps 100
            done
        done
    done
done