# Call, Reward, Repeat: Advancing Dialog State Tracking with GRPO and Function Calling

This repository contains the code for the paper "Call, Reward, Repeat: Advancing Dialog State Tracking with GRPO and Function Calling".

<img src="fig/map_compare.png" width="50%">

Most of the code in this repository was inherited from [FnCTOD](https://github.com/facebookresearch/FnCTOD/tree/main), and we are very grateful for their foundational work.


## Reproduction:

### Install requirements
```bash
pip install -r requirements.txt
```

### Download raw data

See [Data readme](data/README.md)


### Preprocessing and prompting for llama and qwen models

```bash
cd scripts/pretrain_preprocessing/

sh processing-taskmaster.sh
sh processing-woz.sh
sh processing-camres676.sh
sh processing-sgd.sh
sh processing-mse2e.sh


sh prompting-taskmaster.sh
sh prompting-woz.sh
sh prompting-camres676.sh
sh prompting-sgd.sh
sh prompting-mse2e.sh

sh create_finetune_data.sh
```


### Run the scripts
```bash
cd scripts/e2e

chmod +x sft_n_eval.sh
chmod +x grpo_n_eval.sh
chmod +x sft_grpo_n_eval.sh
chmod +x zero-shot.sh
```

- `bash sft_n_eval.sh` - for SFT training and evaluation
- `bash grpo_n_eval.sh` - for GRPO training and evaluation
- `bash sft_grpo_n_eval.sh` - for SFT-GRPO training and evaluation
- `bash zero-shot.sh` - for zero-shot evaluation


### Prerequirements:
- Python 3.10.12
- CUDA Version: 12.6; H100 80GB; NVIDIA Release 24.08 (build 107063150)