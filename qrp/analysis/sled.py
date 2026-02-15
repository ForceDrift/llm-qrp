import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


class SLED_DecodedLLM_GSM8K:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.model, self.tokenizer = self.load_model(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.layers = self.config.num_hidden_layers
        self.lm_head = self.model.get_output_embeddings()

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
        kl_values = []
        for i in range(1, len(output)):
            logits_prev = self.model(
                inputs_embeds=output[i-1]
            ).logits
            logits_curr = self.model(
                inputs_embeds=output[i]
            ).logits

            prev_prob = torch.softmax(logits_prev, dim=-1)
            curr_log_prob = torch.log_softmax(logits_curr, dim=-1)
            kl = torch.nn.functional.kl_div(
                curr_log_prob, prev_prob, reduction="batchmean"
            )
            print(f"layer {i} and layer {i-1}, kl-div: {kl}")
            kl_values.append(kl)

        return kl_values

    # def kl_to_final(self, prompt)


if __name__ == "__main__":

    model_name = "EleutherAI/gpt-neo-1.3B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_model = SLED_DecodedLLM_GSM8K(model_name, device)

    states = test_model.kl_between_current_prev("""

I have a complex logistics problem involving the following parameters:
Inventory: 850 pallets of medical supplies (300 are temperature-sensitive vaccines, 550 are standard PPE).
Origin: A single central port in Savannah, GA.
Destinations: 3 Regional Hubs:
Hub A (Atlanta): High demand, 250 miles away.
Hub B (Charlotte): Medium demand, 320 miles away.
Hub C (Orlando): Critical shortage, 450 miles away.
Timeline: All goods must arrive within a 36-hour window.
Assets: You have access to 15 refrigerated trucks (max 30 pallets each) and 10 standard dry vans (max 40 pallets each).
The Conflict: A major storm is expected in 12 hours affecting the route to Orlando (Hub C), and you currently have a 20% driver shortage (only 20 drivers are available for the 25 total vehicles).
Before giving me a final solution, think step-by-step to outline the constraints, propose two different routing strategies, evaluate the risks of each, and then recommend the best option.""")

    print(f"Final KL list: {states}")
