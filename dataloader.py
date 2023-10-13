from datasets import load_dataset


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
    text = f"### CONTEXT\n{example['instruction'][i]}\n### Question: {example['input'][i]}\n ### Answer: {example['output'][i]}"
    output_texts.append(text)
  return output_texts