# https://medium.com/@jain.sm/finetuning-llama-2-0-on-colab-with-1-gpu-7ea73a8d3db9


import torch
import argparse
import time
import torch.nn as nn
# import bitsandbytes as bnb
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
import os
import transformers
from datasets import load_dataset
from typing import Dict, List, Literal, Optional, Tuple
from pathlib import Path
from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training
from model import get_model
import lightning as L
import bitsandbytes as bnb


from dataloader import create_prompt, get_longest_seq_length, get_batch, load_dataloader_all
from utils import print_trainable_parameters
from losses import chunked_cross_entropy

import json
import pickle

# os.environ["CUDA_VISIBLE_DEVICES"]="7"
torch.cuda.is_available()

# pip install git+https://github.com/huggingface/transformers
torch.set_float32_matmul_precision('high')
def parse_args():
    parser = argparse.ArgumentParser(description="Mistral training")
    parser.add_argument('--dataset', type=str, default="/llm_opt_neurips/datasets/synthetic/v2/raw_data", help="dataset path")
    parser.add_argument('--model', type=str, default="/tampm2/lit-gpt/checkpoints/meta-llama/Llama-2-7b/", help="model name or path")
    parser.add_argument('--output', type=str, default="./trl_results_v2/", help="output_dir path")
    parser.add_argument('--max_length', type=int, default=2048, help="max_seq_length")

    return parser.parse_args()

def train(
        model,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        train_data: List[Dict],
        out_dir,max_iters,
        warmup_steps,learning_rate,
        gradient_accumulation_iters,
        micro_batch_size,
        save_interval,
        max_seq_length,
        fabric
    ) -> None:
    model = fabric.setup_module(model)
    optimizer = fabric.setup_optimizers(optimizer)
    fabric.seed_everything(1337)

    longest_seq_length, longest_seq_ix = get_longest_seq_length(train_data)
    model.max_seq_length = longest_seq_length
    print(
        f"The longest sequence length in the train data is {longest_seq_length}, the model's maximum sequence length is"
        f" {model.max_seq_length}"
    )

    step_count = 0
    total_lengths = 0
    total_t0 = time.perf_counter()

    for iter_num in range(max_iters):
        if step_count <= warmup_steps:
            # linear warmup
            lr = learning_rate * step_count / warmup_steps
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        iter_t0 = time.perf_counter()

        input_ids, targets = get_batch(micro_batch_size, train_data, longest_seq_ix if iter_num == 0 else None,max_seq_length)
        print("input_ids", input_ids)
        print("input_ids shape", input_ids.shape)

        is_accumulating = (iter_num + 1) % gradient_accumulation_iters != 0
        # with fabric.no_backward_sync(model, enabled=is_accumulating):
        # logits = model(input_ids, lm_head_chunk_size=128)
        input_ids = input_ids.to(model.device)
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            # print(model.device)
            # print(input_ids.device)
            loss = model(input_ids)["loss"]
            # import pdb
            # pdb.set_trace()
            # print(logits)
            # print(len(logits))
            # print(logits[0].size())
            # print(targets.size())
            # shift the targets such that output n predicts token n+1
            # logits[0] = logits[0][..., 1:]

            # logits[-1] = logits[-1][..., :-1, :]
            # loss = chunked_cross_entropy(logits, targets[..., 1:].to(model.device),128)

            # loss = chunked_cross_entropy(logits[0][...,1:], targets[..., 1:].to(model.device))
            # loss = chunked_cross_entropy(logits, targets.to(model.device),0)
            # loss = chunked_cross_entropy(logits[0], targets.to(model.device),0)
            # print(loss)
            fabric.backward(loss / gradient_accumulation_iters)
            # loss.backward()

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            if step_count > warmup_steps:
                scheduler.step()
            step_count += 1

        t1 = time.perf_counter()
        total_lengths += input_ids.size(1)

        # if iter_num % (save_interval//5) == 0:
        if iter_num % (warmup_steps) == 0:
            print(
                f"iter {iter_num} step {step_count}: loss {loss.item():.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            )


        if not is_accumulating and step_count % save_interval == 0:
            # checkpoint_path = out_dir / f"iter-{iter_num:06d}-ckpt.pth"
            model.save_pretrained(out_dir)
"""
LLama2
trainable params: 1048576 || all params: 3501461504 || trainable%: 0.029946809319540645
Mistral
trainable params: 851968 || all params: 3752923136 || trainable%: 0.02270145081916221

"""
if __name__ == "__main__":
    args = parse_args()
    ########################################################################
    ################################ CONFIGURATION #########################
    model_name = args.model
    dataset_name = args.dataset
    output_dir = args.output
    # max_seq_length = 2048
    max_seq_length = args.max_length
    learning_rate = 1e-4
    weight_decay = 0.01
    max_iters = 150000
    batch_size = 16
    micro_batch_size = 1
    warmup_steps = 500
    save_interval = 50000
    lora_alpha = 16
    lora_dropout = 0.05
    lora_r = 16
    use_4bit = True
    use_nested_quant = False
    bnb_4bit_compute_dtype = "bfloat16"
    bnb_4bit_quant_type = "nf4"
    # model, tokenizer = get_model("/home/ubuntu/workspace/tampm2/lit-gpt/checkpoints/mistralai/Mistral-7B-Instruct-v0.1")
    # model, tokenizer = get_model("mistralai/Mistral-7B-v0.1")
    # data_path = "/home/ubuntu/workspace/tampm2/lit-gpt/datasets/GAIR/lima"
    # data_path = "/lustre/scratch/client/scratch/llm_opt_neurips/datasets/helm/gsm"
    # data_path = "/lustre/scratch/client/scratch/llm_opt_neurips/datasets/synthetic/v2/raw_data"
    qa_dataset = load_dataloader_all(dataset_name)
    # print(longest_seq_length)
    # print(longest_seq_ix)
    # print(qa_dataset)
    # exit()
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj","v_proj",
                        "k_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                        # "mlp",
                        "lm_head"
                        ],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=use_nested_quant,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
    )
    model, tokenizer = get_model(bnb_config, model_name)
    print_trainable_parameters(model)

    model = prepare_model_for_kbit_training(model)



    model = get_peft_model(model, config)
    print(model.num_parameters())
    print_trainable_parameters(model)

    pkl_train_file = "train_data.pkl"
    if os.path.exists(pkl_train_file):
        # pickle.dump(train_data, open("train_data.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        train_data = pickle.load(open(pkl_train_file, "rb"))
        # train_data = json.load(open("train_data.json", "r"))
    else:
        train_data = qa_dataset.map(lambda samples: {"input_ids":tokenizer.encode(samples['instruction']+samples['input'],max_length=max_seq_length), "labels": tokenizer.encode(samples['output'],max_length=max_seq_length)})
        # json.dump(open("train_data.json", "w"), train_data)
        pickle.dump(train_data, open(pkl_train_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    longest_seq_length, longest_seq_ix = get_longest_seq_length(train_data)
    # print(longest_seq_length)
    # print(longest_seq_ix)
    print(
        f"The longest sequence length in the train data is {longest_seq_length}, at {longest_seq_ix}"
    )
    # train_data = qa_dataset.map(lambda samples: {"input_ids":tokenizer.encode(samples['instruction']+samples['input'],max_length=2048), "labels": tokenizer.encode(samples['instruction']+samples['input']+samples['output'],max_length=2048)})
    train_data = train_data
    # from datasets import load_dataset
    # data = load_dataset("Abirate/english_quotes")
    # mapped_qa_dataset = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

    tokenizer.pad_token = tokenizer.eos_token
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # if False:
    # import bitsandbytes as bnb
    # optimizer = bnb.optim.PagedAdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    optimizer = bnb.optim.Adam8bit(
                        trainable_params,
                        lr=learning_rate, weight_decay=weight_decay
                    )
    # else:
    # optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters // batch_size)
    gradient_accumulation_iters = batch_size // micro_batch_size
    # accelerator = Accelerator(mixed_precision="bf16")
    # model,optimizer, scheduler = accelerator.prepare(model,optimizer, scheduler)
    print_trainable_parameters(model)
    print(model)
    strategy = "auto"
    devices = 1
    precision = "bf16-true"
    plugins = None

    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, plugins=plugins)

    train(
        model,
        optimizer,
        scheduler,
        train_data,
        output_dir, max_iters,
       warmup_steps, learning_rate,
       gradient_accumulation_iters,
       micro_batch_size,
       save_interval,
        max_seq_length,
        fabric
    )




    """
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
    """

