
# unsloth/Meta-Llama-3.1-8B-Instruct
# Qwen/Qwen2.5-3B-Instruct

# declare -a pd_sizes=(25 50 100 200 400)
declare -a pd_sizes=(999999)


declare -a templates=("qwen" "llama31" "qwen3")
declare -a datasets_tuples=(
    # "mwoz multiwoz2.2_-1"
    # "sgd sgd"

)

cd ../..


for ds_tuple in "${datasets_tuples[@]}";do
    read -r accr dataset <<<"$ds_tuple"
    for pd_size in "${pd_sizes[@]}"; do
        for template in "${templates[@]}"; do

    fname=./data/finetunedata/configs/sft-$template-pd$pd_size-$accr.yml
    echo """
    - dataset: $dataset
      split: train
      size: 99999999
      template: $template
      all_turn: false
    """ > $fname

    python create_finetunedata.py --configfile $fname \
                        --template $template \
                        --outputfile ./data/finetunedata/sft/$template-pd$pd_size-$accr.json \
                        --domain_size $pd_size \
                        --max_len 4096
        done
    done
done


# for pd_size in "${pd_sizes[@]}"; do
#     for template in "${templates[@]}"; do

# fname=./data/finetunedata/configs/sft-$template-pd$pd_size-mwoz.yml
# echo """
# - dataset: multiwoz2.1_-1
#   split: train
#   size: $pd_size
#   template: $template
#   all_turn: false
#   all_funcs: true
# """ > $fname

# python create_finetunedata.py --configfile $fname \
#                     --template $template \
#                     --outputfile ./data/finetunedata/sft/$template-pd$pd_size-mwoz.json \
#                     --domain_size $pd_size \
#                     --max_len 4096
# done
# done

# for pd_size in "${pd_sizes[@]}"; do
#     for template in "${templates[@]}"; do

# fname=./data/finetunedata/configs/sft-$template-pd$pd_size-swoz.yml
# echo """
# - dataset: spokenwoz_-1
#   split: train
#   size: $pd_size
#   template: $template
#   all_turn: false
#   all_funcs: true
# """ > $fname

# python create_finetunedata.py --configfile $fname \
#                     --template $template \
#                     --outputfile ./data/finetunedata/sft/$template-pd$pd_size-swoz.json \
#                     --domain_size $pd_size \
#                     --max_len 4096
# done
# done


