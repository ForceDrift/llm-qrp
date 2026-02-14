import torch
from transformers import AutoModelForCausalLM
from transformers.models import AutoConfig
from transformers import AutoTokenizer


class SLED_DecodedLLM_GSM8K:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.model, self.tokenizer = self.load_model(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.layers = self.config.num_hidden_layers

    def load_model(self, model_name):
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

    def kl_between_current_prev(self, prompt):  # model logits per token)

        input_tkns = self.tokenizer(
            prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(
                **input_tkns, output_hidden_states=True).hidden_states

        # convert all hidden states to same vocab space, convert logits to prob, calc divergence
        return output


        # logits_tkn = model(input_tkns).logits
        # p_logits = torch.softmax(logits_tkn, dim=-1)
if __name__ == "__main__":
    model_name = "sshleifer/tiny-gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_model = SLED_DecodedLLM_GSM8K(model_name, device)
    states = test_model.kl_between_current_prev("Hello world")
