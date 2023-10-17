import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel
from typing import List, Optional, Tuple, Union

from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mistral.modeling_mistral import MistralForCausalLM
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
import torch.nn as nn
from accelerate import init_empty_weights
import bitsandbytes as bnb

from transformers.models.mistral.modeling_mistral import MISTRAL_INPUTS_DOCSTRING,CausalLMOutputWithPast,_CONFIG_FOR_DOC
def replace_8bit_linear(model, threshold=6.0, module_to_not_convert="lm_head"):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_8bit_linear(module, threshold, module_to_not_convert)

        if isinstance(module, nn.Linear) and name != module_to_not_convert:
            with init_empty_weights():
                model._modules[name] = bnb.nn.Linear8bitLt(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    has_fp16_weights=False,
                    threshold=threshold,
                )
    return model
class WrapModel(MistralForCausalLM):
    def __init__(self, config):
        super().__init__(config)

    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    def forward(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        # logits = self.lm_head(hidden_states)
        # logits = logits.float()
        lm_head_chunk_size = 128
        return [self.lm_head(x_i) for x_i in hidden_states.split(lm_head_chunk_size, dim=1)]
        # return logits

class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


def get_model(bnb_config, model_name = "meta-llama/Llama-2-7b-chat-hf"):
    # model_name = "bigscience/bloom-3b"
    # model_name = "meta-llama/Llama-2-7b-chat-hf"
    config = AutoConfig.from_pretrained(model_name)
    print(config.architectures)
    # config.architectures = ["WrapModel"]
    # print(config)
    model = WrapModel.from_pretrained(model_name,config = config,
                                                   torch_dtype=torch.bfloat16,
                                                   device_map='auto',
                                                   quantization_config=bnb_config,
                                               )
    # model = AutoModelForCausalLM.from_config(config,quantization_config=bnb_config).int4()
    # model = WrapModel(config).cuda()
    # model = replace_8bit_linear(model).cuda()
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.bfloat16,
    #     device_map='auto',
    #     # peft_config=lora_config,
    #     quantization_config=bnb_config,
    # )
    # print(model)
    # exit()
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
    # model = WrapMistral(model)
    return model, tokenizer