from datasets import load_dataset
from typing import Dict, List, Literal, Optional, Tuple


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
    text = str(example['instruction'][i])+str(example['input'][i])+ " ####Answer: "+str(example['output'][i])
    # text = f"{example['instruction'][i]}. {example['input'][i]}\n"[:1000-len(example['output'][i])] + f"####Answer: {example['output'][i]}"
    # print(len(text))
    output_texts.append(text)
  return output_texts

def get_longest_seq_length(data: List[Dict]) -> Tuple[int, int]:
  # find out the minimum max_seq_length required during fine-tuning (saves memory!)
  lengths = [len(d["instruction"]) + len(d["input"]) + len(d["output"]) for d in data]
  longest_seq_length = max(lengths)
  longest_seq_ix = lengths.index(longest_seq_length)
  return longest_seq_length, longest_seq_ix
