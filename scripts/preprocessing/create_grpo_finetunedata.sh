cd ../..


declare -a pd_sizes=(25 50 100 200 400)
declare -a templates=("qwen" "llama31")
declare -a datasets_tuples=(
    "mwoz multiwoz2.2_-1" 
    "sgd sgd"
)


for ds_tuple in "${datasets_tuples[@]}";do
    read -r ds full_ds_name <<<"$ds_tuple"
    for pd_size in "${pd_sizes[@]}"; do
        for template in "${templates[@]}"; do
            python create_grpo_finetunedata.py --input ./data/finetunedata/sft/$template-pd$pd_size-$ds.json \
                    --output ./data/finetunedata/grpo/$template-pd$pd_size-$ds.json
        done
    done
done

