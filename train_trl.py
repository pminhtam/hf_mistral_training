import json
import re
from pprint import pprint

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import notebook_login
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM  # For supervised finetuning

from dataloader import load_dataloader, new_formatting_prompts_func, load_dataloader_all
import argparse, os


def get_lora_layer_config(lora_query=True, lora_key=False, lora_value=True, lora_projection=False, lora_mlp=False,
                          lora_head=False):
    configs = []
    if lora_query:
        configs.append('q_proj')
    if lora_key:
        configs.append('k_proj')
    if lora_value:
        configs.append('v_proj')
    if lora_projection:
        configs.append('out_proj')
    if lora_mlp:
        configs.append('gate_proj')
        configs.append("up_proj")
        configs.append("down_proj")
    if lora_head:
        configs.append('lm_head')

    return configs
def new_formatting_prompts_func_collator(example):
    # output_texts = []
    # for i in range(len(example['instruction'])):
    output_texts = [str(inst) + str(inp) + "  ###Answer: " + str(out) for inst, inp, out in zip(example['instruction'], example['input'], example['output'])]
        # output_texts.append(text)
    return output_texts

def get_args():
    parser = argparse.ArgumentParser(description='Take hyperparameter')
    parser.add_argument('--data', type=str, help='data path', default='data')
    parser.add_argument('--lr', type=float, help='initial learning rate', default=5e-6)
    parser.add_argument('--lr_scheduler', type=str, help='learning rate scheduler', default='constant')
    parser.add_argument('--epochs', type=int, help='number of training epochs', default=2)
    parser.add_argument('--neftune', type=int, help='neftune noise integer value', default=5)
    parser.add_argument('--exp_name', type=str, help='experiment name and path to save checkpoint', default='default')
    parser.add_argument('--lora_r', type=int, help='LoRA rank', default=64)
    parser.add_argument('--warmup_step', type=int, help='warmup step', default=0)
    parser.add_argument('--warmup_ratio', type=float, help='warmup ratio', default=0.0)
    parser.add_argument('--batch_size', type=int, help='global batch size', default=16)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    ################################################################################
    # Tunable hyperaparameter
    ################################################################################

    args = get_args()
    if not os.path.exists(f'train_ckpt/{args.exp_name}'):
        os.makedirs(f'train_ckpt/{args.exp_name}')
    with open(f'train_ckpt/{args.exp_name}/args.txt', 'w') as fout:
        fout.write(str(args))

    data_path = args.data
    # Initial learning rate (AdamW optimizer)
    learning_rate = args.lr
    # Learning rate schedule (constant a bit better than cosine)
    lr_scheduler_type = args.lr_scheduler
    # Number of training epochs
    num_train_epochs = args.epochs
    # neftune noise
    neftune_noises = args.neftune
    # LoRA attention dimension
    lora_r = args.lora_r
    # Output directory where the model predictions and checkpoints will be stored
    output_dir = f"./train_ckpt/{args.exp_name}"
    # save final trained lora model
    trained_model_path = f'train_ckpt/{args.exp_name}/saved'
    # Ratio of steps for a linear warmup (from 0 to learning rate)
    warmup_ratio = args.warmup_ratio
    warmup_step = args.warmup_step
    # Batch size per GPU for training
    batch_size = args.batch_size
    per_device_train_batch_size = 1

    # Number of update steps to accumulate the gradients for
    gradient_accumulation_steps = batch_size // per_device_train_batch_size

    model_name = "ckpt/mistralai/Mistral-7B-v0.1"

    print("Proceeding dataset ...")
    new_data = load_dataloader_all(data_path)
    # train_dataset = new_data.map(new_formatting_prompts_func, batched=True, remove_columns=new_data.column_names)

    print("Setting up configuration for quantization")
    ################################################################################
    # QLoRA parameters
    ################################################################################

    # Alpha parameter for LoRA scaling
    lora_alpha = 32

    # Dropout probability for LoRA layers
    lora_dropout = 0.05

    lora_query = True
    lora_key = True
    lora_value = True
    lora_projection = False
    lora_mlp = True
    lora_head = False

    ################################################################################
    # bitsandbytes parameters
    ################################################################################

    # Activate 4-bit precision base model loading
    use_4bit = True

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"

    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = False

    ################################################################################
    # TrainingArguments parameters
    ################################################################################

    # Enable fp16/bf16 training (set bf16 to True with an A100)
    fp16 = False
    bf16 = True

    # Enable gradient checkpointing
    gradient_checkpointing = True

    # Maximum gradient normal (gradient clipping)
    max_grad_norm = 0.3

    # Weight decay to apply to all layers except bias/LayerNorm weights
    weight_decay = 0.01

    # Optimizer to use
    optim = "paged_adamw_8bit"

    # Number of training steps (overrides num_train_epochs)
    max_steps = -1

    # Group sequences into batches with same length
    # Saves memory and speeds up training considerably
    group_by_length = True

    # Save checkpoint every X updates steps
    save_steps = 1000

    # Log every X updates steps
    logging_steps = 1000

    ################################################################################
    # SFT parameters
    ################################################################################

    # Maximum sequence length to use
    max_seq_length = 1300

    # Pack multiple short examples in the same input sequence to increase efficiency
    packing = False

    # Load the entire model on the GPU 0
    device_map = {"": 0}

    # Load the base model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    print('Loading model ...')
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map
    )

    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1
    base_model.config.window = 256

    # Load MistralAI tokenizer
    print('Loading tokenizer ...')
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load LoRA configuration
    print('Setting up LoRA config ...')
    target_modules = get_lora_layer_config(lora_query, lora_key, lora_value, lora_projection, lora_mlp, lora_head)
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Set training parameters
    print('Setting up training configurations ...')
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,  # the number of training steps the model will take
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_step,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        save_total_limit=2,
    )
    response_template = "###Answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # Set supervised fine-tuning parameters
    print('Initialize trainer ...')
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=new_data,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
        neftune_noise_alpha=neftune_noises,

        formatting_func=new_formatting_prompts_func_collator,
        data_collator=collator,
    )

    trainer.train()

    trainer.model.save_pretrained(trained_model_path)
