cd ../..

for pd_size in 200
do
    for template in qwen llama31
    do
    python create_grpo_finetunedata.py --input ./data/finetunedata/sft-swoz-$template-pd$pd_size.json \
                    --output ./data/finetunedata/grpo-swoz-$template-pd$pd_size.json 
    done
done