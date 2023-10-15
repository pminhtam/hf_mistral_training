# https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da


import torch
import argparse

import torch.nn as nn
# import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback
import os
import transformers

from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training
from model import get_model
from dataloader import load_dataloader, new_formatting_prompts_func
from utils import print_trainable_parameters
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
# os.environ["CUDA_VISIBLE_DEVICES"]="7"
torch.cuda.is_available()

def get_lora_layer_config(lora_query = True, lora_key = False, lora_value = True, lora_projection = False, lora_mlp = False, lora_head = False):
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

class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))

# pip install git+https://github.com/huggingface/transformers
def parse_args():
    parser = argparse.ArgumentParser(description="Mistral training")
    parser.add_argument('--dataset', type=str, default="data", help="dataset path")
    parser.add_argument('--output', type=str, default="./trl_results_v2/", help="output_dir path")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    ########################################################################
    ################################ CONFIGURATION #########################
    batch_size = 128
    micro_batch_size = 1
    per_device_eval_batch_size = 1
    gradient_accumulation_steps = batch_size // micro_batch_size
    learning_rate = 5e-5
    warmup_ratio = 0.1
    max_grad_norm = 0.3
    weight_decay = 0.01
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    lora_query = True
    lora_key = False
    lora_value = True
    lora_projection = False
    lora_mlp = False
    lora_head = False

    max_seq_length = 512
    model_name = "mistralai/Mistral-7B-v0.1"
    dataset_name = args.dataset
    use_4bit = False
    use_nested_quant = False
    bnb_4bit_compute_dtype = "float16"
    bnb_4bit_quant_type = "nf4-dq"
    # num_train_epochs = 10
    fp16 = False
    bf16 = True
    # packing = False
    # gradient_checkpointing = True
    # optim = "paged_adamw_32bit"
    optim = "adamw_hf"
    lr_scheduler_type = "cosine"
    max_steps = 100
    # group_by_length = True
    logging_steps = 100
    save_steps = 100
    output_dir = "./trl_results/"


    ########################################################################


    # model, tokenizer = get_model("meta-llama/Llama-2-7b-chat-hf")
    # model, tokenizer = get_model("/home/ubuntu/workspace/tampm2/lit-gpt/checkpoints/mistralai/Mistral-7B-Instruct-v0.1")
    # model, tokenizer = get_model("mistralai/Mistral-7B-v0.1")

    print("Proceeding dataset ...")
    data_path = dataset_name
    new_data = load_dataloader(data_path)
    dataset = new_data.map(new_formatting_prompts_func, batched=True, remove_columns=new_data.column_names)
    # compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=use_4bit,
    #     bnb_4bit_quant_type=bnb_4bit_quant_type,
    #     bnb_4bit_compute_dtype=compute_dtype,
    #     bnb_4bit_use_double_quant=use_nested_quant,
    # )
    # if compute_dtype == torch.float16 and use_4bit:
    #     major, _ = torch.cuda.get_device_capability()
    #     if major >= 8:
    #         print("=" * 80)
    #         print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
    #         print("=" * 80)
    # Load the entire model on the GPU 0
    # switch to `device_map = "auto"` for multi-GPU
    # device_map = "auto"
    # device_map = "balanced"
    # print('Loading model ...')
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     # device_map='auto',
    #     device_map=None,
    #     # torch_dtype=torch.bfloat16,
    #     torch_dtype=None,
    # )
    # check: https://github.com/huggingface/transformers/pull/24906
    # model.config.pretraining_tp = 1

    target_modules = get_lora_layer_config(lora_query, lora_key, lora_value, lora_projection, lora_mlp, lora_head)
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # Fix weird overflow issue with fp16 training
    tokenizer.padding_side = "right"
    # model.config.use_cache = False # silence the warnings. Please re-enable for inference!
    # print_trainable_parameters(model)

    print("Initialized training arguments ...")
    training_arguments = transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        bf16=bf16,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        save_total_limit=2,
        max_grad_norm=max_grad_norm,
    )

    print('Training with SFT trainer ...')
    # https://huggingface.co/docs/trl/main/en/sft_trainer#advanced-usage
    trainer = SFTTrainer(
        args=training_arguments,
        model=model_name,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        callbacks=[PeftSavingCallback()],
        tokenizer=tokenizer,
        packing=True,
    )
    trainer.train()
    trainer.save_model(output_dir)

