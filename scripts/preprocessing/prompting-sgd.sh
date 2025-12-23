# sgd

cd ../../


echo "preprocessing data"
python -m src.sgd.preprocess

echo "create splits and prompting"
for split in train test
do
    for template in llama31 qwen
    do
        for all_turn in False
        do
            CUDA_VISIBLE_DEVICES=$devices python -m src.sgd.prompting \
                                                    --split $split \
                                                    --template $template \
                                                    --all_turn $all_turn
        done
    done
done
