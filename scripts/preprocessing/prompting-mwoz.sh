# multiwoz
cd ../..
dataset_version=2.1
n_sample=-1
# for split in train test
# do
#     for template in qwen llama31
#     do
#         for all_turn in False
#         do
#         python -m src.multiwoz.prompting \
#                 --dataset_version $dataset_version \
#                 --split $split \
#                 --n_sample $n_sample \
#                 --template $template \
#                 --all_turn $all_turn \
#             done
#         done
#     done
# done

for template in qwen llama31; do
    for split in train test; do
        python -m src.spokenwoz.prompting \
            --all_turn $all_turn \
            --split $split \
            --n_sample $n_sample \
            --template $template
    done
done






