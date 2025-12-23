cd ../..

declare -a datasets=("sft_pretrain")

declare -a pd_sizes=(1 10 20 50 100 200 300 400)

declare -a dataset_seeds=(1234 2345 3456 4567 5678)

declare -a lora_targets=('[q_proj,k_proj,v_proj]')
declare -a lora_rs=(16)
declare -a epochs=(1)

declare -a tuples=(
    "xlam Salesforce/xLAM-2-1b-fc-r"
    "xlam Salesforce/xLAM-2-3b-fc-r"
    "xlam Salesforce/Llama-xLAM-2-8b-fc-r"
    "xlam Salesforce/xLAM-2-32b-fc-r"

    "qwen Qwen/Qwen2.5-1.5B-Instruct"
    "qwen Qwen/Qwen2.5-3B-Instruct"
    "qwen Qwen/Qwen2.5-7B-Instruct"
    "qwen Qwen/Qwen2.5-14B-Instruct"
    "qwen Qwen/Qwen2.5-32B-Instruct"

    "llama31 unsloth/Meta-Llama-3.1-8B-Instruct"
    )



devices=0
save_dir="DST_scores"
queue=""

batch_size=8
micro_batch_size=2
cutoff_len=4096
with_unsloth=True
add_eos_token=False
lr=0.0003
train_on_response=False
save_merged=True
n_eval=-1

eval_datasets="mwoz24;mwoz22;mwoz21"
multi_domain=True
ref_domain=True
ref_bs=False
add_prev=True
dst_nshot=0
nlg_nshot=0
function_type="json"
split="test"
handler=trained
temperature=0.001
top_p=0.2
top_k=-1

echo "Fit"
for dataset_seed in "${dataset_seeds[@]}"; do
    for lora_r in "${lora_rs[@]}"; do
        for targets in "${lora_targets[@]}"; do
            for ep in "${epochs[@]}"; do
                for pd_size in "${pd_sizes[@]}"; do
                    for ds in "${datasets[@]}"; do
                        for tuple in "${tuples[@]}"; do
                            read -r template base_model <<<"$tuple"

                            datafile=$template-pd-1

                            python sft_and_eval.py \
                            --pd_size $pd_size \
                            --dataset_seed $dataset_seed \
                            --base_model $base_model \
                            --train_dataset $ds \
                            --data-path ./data/finetunedata/sft_pretrain/$datafile.json \
                            --output_dir ./ckpt/lora_ckpt/SFT/$base_model-$datafile-ep$ep-tr-on-resp$train_on_response \
                            --template $template \
                            --batch_size $batch_size \
                            --micro_batch_size $micro_batch_size \
                            --num_epochs $ep \
                            --learning_rate $lr \
                            --cutoff_len $cutoff_len \
                            --with_unsloth $with_unsloth \
                            --val_set_size 0 \
                            --lora_r $lora_r \
                            --lora_alpha $lora_r \
                            --lora_dropout 0.05 \
                            --lora_target_modules $targets  \
                            --train_on_response $train_on_response \
                            --add_eos_token $add_eos_token \
                            --group_by_length False \
                            --lr_scheduler 'cosine' \
                            --warmup_steps 0 \
                            --save_merged $save_merged \
                            --eval_datasets $eval_datasets \
                            --top_p $top_p \
                            --split $split \
                            --n_eval $n_eval \
                            --dst_nshot $dst_nshot \
                            --nlg_nshot $nlg_nshot \
                            --add_prev $add_prev \
                            --ref_domain $ref_domain \
                            --ref_bs $ref_bs \
                            --multi_domain $multi_domain \
                            --function_type $function_type \
                            --generate \
                            --eval_template $template \
                            --handler trained \
                            --gpu-memory-utilization 0.9 \
                            --queue $queue \
                            --save_dir $save_dir

                        done
                    done
                done
            done
        done
    done
done