import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer


if __name__ == '__main__':
    model_name = "ckpt/mistralai/Mistral-7B-Instruct-v0.1/"
    adapters_name = "./finetuned_model"

    print(f"Starting to load the model {model_name} into memory")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        #load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # model = PeftModel.from_pretrained(model, adapters_name)
    # model = model.merge_and_unload()
    # tok = LlamaTokenizer.from_pretrained(model_name)
    # tok.bos_token_id = 1
    # stop_token_ids = [0]

    print(f"Successfully loaded the model {model_name} into memory")

    context = ""
    question = "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?"
    batch = tokenizer(f"### CONTEXT\n{context}\n\n### QUESTION\n{question}\n\n### ANSWER\n", return_tensors='pt')
    batch = batch.to("cuda")
    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**batch, max_new_tokens=200)
    answer = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print(answer)