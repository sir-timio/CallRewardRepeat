# mse2e

cd ../..

for split in train
do
    for template in llama31 qwen
    do
        for all_turn in False
        do
            CUDA_VISIBLE_DEVICES=$devices python -m src.mse2e.prompting \
                                                    --split $split \
                                                    --template $template \
                                                    --all_turn $all_turn
        done
    done
done
