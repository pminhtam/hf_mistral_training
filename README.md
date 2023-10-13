# Train mistral with huggingface 


## Train 

Train with `transformers.Trainer`

```shell
python train.py
```

Train with `trl`


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

