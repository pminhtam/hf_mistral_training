from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from peft import LoraConfig, PeftModel

if __name__ == "__main__":
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map='auto',
    )

    trained_model_path = 'train_model/test/'
    merged_model= PeftModel.from_pretrained(base_model, trained_model_path)
    merged_model= merged_model.merge_and_unload()

    # Save the merged model
    merged_model.save_pretrained("merged_model",safe_serialization=True)
    # tokenizer.save_pretrained("merged_model")
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"