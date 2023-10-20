import torch
import glob
from datasets import load_dataset, Dataset
from typing import Dict, List, Literal, Optional, Tuple
import json

def load_dataloader(data_path):
  """
  Read JSONL -> convert HF datasets
  """
  data = []
  domains = ['chat', 'cnn', 'complexqa', 'complexqa1', 'dolly', 'math', 'science']
  for domain in domains:
    with open(f'{data_path}/mistral_{domain}_prompt_data.jsonl') as fin:
      for line in fin:
        _data = json.loads(line)
        data.append({'instruction': str(_data['instruction']), 'input': str(_data['input']), 'output': str(_data['output'])})
  dataset = Dataset.from_list(data)
  return dataset

def create_prompt(context, question, answer):
  if len(answer["text"]) < 1:
    answer = "Cannot Find Answer"
  else:
    answer = answer["text"][0]
  prompt_template = f"### CONTEXT\n{context}\n\n### QUESTION\n{question}\n\n### ANSWER\n{answer}</s>"
  return prompt_template

def formatting_prompts_func(example):
  output_texts = []
  for i in range(len(example['input'])):
    # text = f"### CONTEXT\n{example['instruction'][i]}\n### Question: {example['input'][i]}\n ### Answer: {example['output'][i]}"
    # text = f"{example['instruction'][i]}. {example['input'][i]}\n ####Answer: {example['output'][i]}"
    # text = "####CONTEXT: " + str(example['instruction'][i])+str(example['input'][i])+ " ####Answer: "+str(example['output'][i])
    text = str(example['instruction'][i])+str(example['input'][i])+ " #Answer: "+str(example['output'][i])
    # text = f"{example['instruction'][i]}. {example['input'][i]}\n"[:1000-len(example['output'][i])] + f"####Answer: {example['output'][i]}"
    # print(len(text))
    output_texts.append(text)
  return output_texts

def new_formatting_prompts_func(example):
  example['text'] = [str(inst) + str(inp) + str(out) for inst, inp, out in zip(example['instruction'], example['input'], example['output'])]
  return example

def get_longest_seq_length(data: List[Dict]) -> Tuple[int, int]:
  # find out the minimum max_seq_length required during fine-tuning (saves memory!)
  lengths = [len(d["input_ids"]) for d in data]
  longest_seq_length = max(lengths)
  longest_seq_ix = lengths.index(longest_seq_length)
  return longest_seq_length, longest_seq_ix

def load_dataloader_all(data_path):
  """
  Read JSONL -> convert HF datasets
  """
  data = []
  domains = ['chat', 'cnn', 'complexqa', 'complexqa1', 'dolly', 'math', 'science']
  for path in glob.glob(f'{data_path}/*.jsonl'):
    with open(path) as fin:
      for line in fin:
        _data = json.loads(line)
        data.append({'instruction': str(_data['instruction']), 'input': str(_data['input']), 'output': str(_data['output'])})
  dataset = Dataset.from_list(data)
  return dataset

def get_batch(
      micro_batch_size, data: List[Dict], longest_seq_ix: Optional[int] = None, max_seq_length = 1024
) -> Tuple[torch.Tensor, torch.Tensor]:
  ix = torch.randint(len(data), (micro_batch_size,))
  if longest_seq_ix is not None:
    # force the longest sample at the beginning so potential OOMs happen right away
    ix[0] = longest_seq_ix
  # print(data)
  # print(data[1000])
  # print(data[90000])
  # print(ix)
  # print(ix[0])
  # print(data[int(ix[0])])
  # print(data[ix[0]])
  # print(data[i] for i in ix)
  input_ids = [torch.tensor(data[int(i)]["input_ids"]).type(torch.int64) for i in ix]
  labels = [torch.tensor(data[int(i)]["labels"]).type(torch.int64) for i in ix]
  # print(input_ids, labels)
  # this could be `longest_seq_length` to have a fixed size for all batches
  max_len_ids = max(len(s) for s in input_ids)
  max_len_label = max(len(s) for s in labels)
  max_len = max(max_len_ids, max_len_label)

  def pad_right(x, pad_id):
    # pad right based on the longest sequence
    n = max_len - len(x)
    # print("57:  ",x.shape)
    x = torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))
    # print("59  : ",x.shape)
    return x

  x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
  # print("63  x.shape   : ",x.shape)
  y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
  # print("65  y.shape  : ",y.shape)
  x = x[,:1000,]
  return x, y

