# https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da


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
from dataloader import create_prompt, formatting_prompts_func
from utils import print_trainable_parameters
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
# os.environ["CUDA_VISIBLE_DEVICES"]="7"
torch.cuda.is_available()

# pip install git+https://github.com/huggingface/transformers
def parse_args():
    parser = argparse.ArgumentParser(description="Mistral training")
    parser.add_argument('--dataset', type=str, default="/llm_opt_neurips/datasets/synthetic/v2/raw_data", help="dataset path")
    parser.add_argument('--output', type=str, default="./trl_results_v2/", help="output_dir path")
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    ########################################################################
    ################################ CONFIGURATION #########################
    per_device_train_batch_size = 4
    per_device_eval_batch_size = 1
    gradient_accumulation_steps = 4
    learning_rate = 2e-4
    max_grad_norm = 0.3
    weight_decay = 0.001
    lora_alpha = 16
    lora_dropout = 0.1
    lora_r = 64
    max_seq_length = 512
    model_name = "mistralai/Mistral-7B-v0.1"
    # dataset_name = "/lustre/scratch/client/scratch/llm_opt_neurips/datasets/helm/gsm"
    # dataset_name = "/llm_opt_neurips/datasets/helm/gsm"
    # dataset_name = "/llm_opt_neurips/datasets/synthetic/v2/raw_data"
    dataset_name = args.dataset
    use_4bit = True
    use_nested_quant = False
    bnb_4bit_compute_dtype = "float16"
    bnb_4bit_quant_type = "nf4"
    num_train_epochs = 10
    fp16 = False
    bf16 = True
    packing = False
    gradient_checkpointing = True
    optim = "paged_adamw_32bit"
    lr_scheduler_type = "constant"
    max_steps = 10000
    warmup_ratio = 0.03
    group_by_length = True
    save_steps = 100
    logging_steps = 100
    output_dir = "./trl_results/"


    ########################################################################


    # model, tokenizer = get_model("meta-llama/Llama-2-7b-chat-hf")
    # model, tokenizer = get_model("/home/ubuntu/workspace/tampm2/lit-gpt/checkpoints/mistralai/Mistral-7B-Instruct-v0.1")
    # model, tokenizer = get_model("mistralai/Mistral-7B-v0.1")

    data_path = dataset_name
    qa_dataset = load_dataset(data_path)
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)
    # Load the entire model on the GPU 0
    # switch to `device_map = "auto"` for multi-GPU
    device_map = {"": 0}
    # device_map = "auto"
    # device_map = "balanced"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        # use_auth_token=True
    )
    # check: https://github.com/huggingface/transformers/pull/24906
    model.config.pretraining_tp = 1

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        target_modules=["q_proj","v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # Fix weird overflow issue with fp16 training
    tokenizer.padding_side = "right"
    model.config.use_cache = False # silence the warnings. Please re-enable for inference!
    # prepare model for training
    # model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    # max_seq_length = tokenizer.model_max_length
    # print("max_seq_length   : ", max_seq_length)
    print_trainable_parameters(model)

    # mapped_qa_dataset = qa_dataset.map(
    #     lambda samples: tokenizer(create_prompt(samples['instruction'] if "instruction" in samples else "", samples['input'], samples['output'])))
    # mapped_qa_dataset = qa_dataset.map(
    #     lambda samples: {"text":create_prompt(samples['instruction'] if "instruction" in samples else "", samples['input'], samples['output'])})
    response_template = "####Answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    """
    instruction_template = "### Human:"
    response_template = "### Assistant:"
    collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template,
                                               response_template=response_template, tokenizer=tokenizer, mlm=False)
    """



    training_arguments = transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
    )
    # https://huggingface.co/docs/trl/main/en/sft_trainer#advanced-usage
    trainer = SFTTrainer(
        model=model,
        train_dataset=qa_dataset["train"],
        formatting_func=formatting_prompts_func,
        peft_config=peft_config,
        # dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        # packing=packing,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(output_dir)

