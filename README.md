# Train mistral with huggingface 


## Train 

[//]: # (Train with `transformers.Trainer`)
Train with dataloader like lit-gpt
```shell
python train.py  --dataset /llm_opt_neurips/datasets/synthetic/v2/raw_data/ --output ./trl_results_v2/ --model ckpt/mistralai/Mistral-7B-Instruct-v0.1/ --max_length 512
```

Train with `trl`
```shell
python train_trl.py  --dataset /llm_opt_neurips/datasets/synthetic/v2/raw_data/ --output ./trl_results_v2
```

## Infer 

Infer with lora weight 
```shell
python infer.py
```

Infer with origin weight

```shell
python infer_origin.py
```


## Requirements

```shell
pip install git+https://github.com/huggingface/transformers
```
datasets
peft
trl
bitsandbytes
scipy

