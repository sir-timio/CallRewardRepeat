cd ../..

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

eval_datasets="mwoz24;mwoz22;mwoz21"
n_eval=-1
cutoff_len=4096
add_eos_token=False # default False
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

echo "Zero shot eval"
    for tuple in "${tuples[@]}"; do
        read -r template base_model <<<"$tuple"

        python eval.py \
        --base_model $base_model \
        --template $template \
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