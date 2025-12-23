cd ../..

for template in llama31 qwen; do
for pd_size in -1; do
    python create_finetunedata.py --configfile ./data/finetunedata/pretrain_configs/$template.yml \
                        --outputfile ./data/finetunedata/sft_pretrain/$template-pd$pd_size.json \
                        --domain_size $pd_size \
                        --max_len 4096 \
                        --template $template
done
done