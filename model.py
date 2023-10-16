import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig


class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


def get_model(bnb_config, model_name = "meta-llama/Llama-2-7b-chat-hf"):
    # model_name = "bigscience/bloom-3b"
    # model_name = "meta-llama/Llama-2-7b-chat-hf"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        # peft_config=lora_config,
        quantization_config=bnb_config,

    )
    # model_id = "meta-llama/Llama-2–7b-chat-hf"
    # bnb_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_use_double_quant=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_compute_dtype=torch.bfloat16
    #         )


    # model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={'auto'})
    # tokenizer = AutoTokenizer.from_pretrained("bigscience/tokenizer")
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2–7b-chat-hf")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # print(model)
    for param in model.parameters():
      param.requires_grad = False  # freeze the model - train adapters later
      # if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        # param.data = param.data.to(torch.float32)

    # model.gradient_checkpointing_enable()  # reduce number of stored activations
    # model.enable_input_require_grads()


    # model.lm_head = CastOutputToFloat(model.lm_head)
    # model.gradient_checkpointing_enable()
    return model, tokenizer