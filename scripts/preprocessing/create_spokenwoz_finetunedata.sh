
cd ../..

for pd_size in 100 200 400
do
python create_finetunedata.py --configfile ./data/finetunedata/configs/sft-llama31-swoz.yml \
                    --tokenizer unsloth/Meta-Llama-3.1-8B-Instruct \
                    --outputfile ./data/finetunedata/sft-swoz-llama31-pd$pd_size.json \
                    --domain_size $pd_size \
                    --max_len 4096
done

for pd_size in 100 200 400
do
python create_finetunedata.py --configfile ./data/finetunedata/configs/sft-qwen-swoz.yml \
                    --tokenizer Qwen/Qwen2.5-3B-Instruct \
                    --outputfile ./data/finetunedata/sft-swoz-qwen-pd$pd_size.json \
                    --domain_size $pd_size \
                    --max_len 4096
done

