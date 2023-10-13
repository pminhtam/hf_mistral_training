# https://medium.com/@jain.sm/finetuning-llama-2-0-on-colab-with-1-gpu-7ea73a8d3db9


import torch
import argparse

import torch.nn as nn
# import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
import os
import transformers
from datasets import load_dataset

from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training
from model import get_model
from dataloader import create_prompt
from utils import print_trainable_parameters
# os.environ["CUDA_VISIBLE_DEVICES"]="7"
torch.cuda.is_available()

# pip install git+https://github.com/huggingface/transformers

def parse_args():
    parser = argparse.ArgumentParser(description="Mistral training")


    return parser.parse_args()

if __name__ == "__main__":
    # model, tokenizer = get_model("meta-llama/Llama-2-7b-chat-hf")
    # model, tokenizer = get_model("/home/ubuntu/workspace/tampm2/lit-gpt/checkpoints/mistralai/Mistral-7B-Instruct-v0.1")
    model, tokenizer = get_model("mistralai/Mistral-7B-v0.1")
    # data_path = "/home/ubuntu/workspace/tampm2/lit-gpt/datasets/GAIR/lima"
    data_path = "/lustre/scratch/client/scratch/llm_opt_neurips/datasets/helm/gsm"
    qa_dataset = load_dataset(data_path)


    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["q_proj","v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)


    mapped_qa_dataset = qa_dataset.map(
        lambda samples: tokenizer(create_prompt(samples['instruction'], samples['input'], samples['output'])))

    # from datasets import load_dataset
    # data = load_dataset("Abirate/english_quotes")
    # mapped_qa_dataset = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

    tokenizer.pad_token = tokenizer.eos_token


    trainer = transformers.Trainer(
        model=model,
        train_dataset=mapped_qa_dataset["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=100,
            max_steps=100,
            learning_rate=1e-3,
            fp16=True,
            logging_steps=1,
            output_dir='outputs',
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
    trainer.save_model("./finetuned_model")

