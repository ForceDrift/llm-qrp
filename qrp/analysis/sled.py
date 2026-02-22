from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch.nn.functional as F
import torch
import numpy as np


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

    def kl_between_current_final(self, prompt):

        input_tkns = self.tokenizer(
            prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(
                **input_tkns, output_hidden_states=True).hidden_states
            logits_final = self.model(
                inputs_embeds=self.lm_head
            ).logits
            final_prob = torch.log_softmax(logits_final, dim=-1)

            kl_values = []

            for i in range(1, len(output)):
                logits_curr = self.model(inputs_embeds=output[i]).logits
                curr_log_prob = torch.log_softmax(logits_curr, dim=-1)
                kl = torch.nn.functional.kl_div(
                    curr_log_prob, final_prob, reduction="batchmean")
                kl_values.append(kl)

        return kl_values

    def prob_final_pred_token(self, prompt):

        input_tkns = self.tokenizer(
            prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model(
                **input_tkns, output_hidden_states=True).hidden_states
        prob = torch.nn.functional.softmax(output, dim=-1)
        tkn_prob, token_id = torch.max(prob, dim=-1)
        pred_token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return (pred_token, tkn_prob)

    # TODO: add token rank and SLED

    # modified from https://github.com/JayZhang42/SLED/blob/main/sled_decoding.py#L155
    # mask certain tokens to limit disagreement computation and reduce noise

    def get_relative_top_filter(self, scores: torch.FloatTensor, relative_top: float = 0.1):
        scores_prob = scores.log_softmax(dim=-1)
        sorted_logits, sorted_idx = torch.sort(scores_prob, descending=True)
        min_thresh = sorted_logits[0]
        probs_max = torch.max(scores_prob, dim=-1).values

        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        return scores_prob < probs_thresh

    def layer_disagreement(self, prompt, evolution_scale=10):
        with torch.no_grad():
            input_tkns = self.tokenizer(
                prompt, return_tensors="pt").input_ids.to(self.device)
            outputs = self.model(**input_tkns, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            premature_layers = []
            for h in hidden_states:
                logits = self.model.lm_head(h)
                premature_layers.append(logits)

            mature_logits = premature_layers[-1]
            seq_len = input_tkns["input_ids"].shape[1]

            disagreement = []

            for seq_i in range(seq_len - 1):
                stacked_premature = torch.stack(
                    [l[:, seq_i, :] for l in premature_layers[:-1]], dim=0)
                mature_token_logits = mature_logits[:, seq_i, :]

                # softmax
                softmax_mature = torch.nn.functional.softmax(
                    mature_token_logits, dim=-1)
                softmax_premature = torch.nn.functional.softmax(
                    stacked_premature, dim=-1)

                topk_prob, topk_indices = torch.topk(
                    softmax_mature, evolution_scale, dim=-1)

                topk_indices = topk_indices[0]

                divergence = stacked_premature - \
                    mature_token_logits.unsqueeze(0)

                candidate_gradients_expanded = softmax_premature.expand(
                    -1, len(topk_indices), -1)
                candidate_mask = torch.zeros_like(
                    candidate_gradients_expanded)
                topk_indices_expanded = topk_indices.unsqueeze(
                    0).unsqueeze(2)
                candidate_mask.scatter_(2, topk_indices_expanded.expand(softmax_premature.size(0), -1, -1),
                                        1)
                candidate_gradients_expanded = candidate_gradients_expanded - candidate_mask
                candidate_gradients_expanded = candidate_gradients_expanded.to(
                    torch.float32)
                layer_divergence_expanded = divergence.to(torch.float32)
                layer_dot_results = F.cosine_similarity(candidate_gradients_expanded, layer_divergence_expanded,
                                                        dim=2)

 #


if __name__ == "__main__":

    model_name = "EleutherAI/gpt-neo-1.3B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_model = SLED_DecodedLLM_GSM8K(model_name, device)

    states = test_model.kl_between_current_prev(
        "You discover a new color that no human has ever seen. How would you describe it to someone?")

    print(f"Final KL list: {states}")
