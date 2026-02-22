import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


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

        input_tkns = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**input_tkns, output_hidden_states=True).hidden_states
        kl_values = []

        for i in range(1, len(output)):
            logits_prev = self.model(inputs_embeds=output[i - 1]).logits
            logits_curr = self.model(inputs_embeds=output[i]).logits

            prev_prob = torch.softmax(logits_prev, dim=-1)
            curr_log_prob = torch.log_softmax(logits_curr, dim=-1)
            kl = torch.nn.functional.kl_div(curr_log_prob, prev_prob, reduction="batchmean")
            print(f"layer {i} and layer {i - 1}, kl-div: {kl}")
            kl_values.append(kl)

        return kl_values

    def kl_between_current_final(self, prompt):

        input_tkns = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**input_tkns, output_hidden_states=True).hidden_states
            logits_final = self.model(inputs_embeds=self.lm_head).logits
            final_prob = torch.log_softmax(logits_final, dim=-1)

            kl_values = []

            for i in range(1, len(output)):
                logits_curr = self.model(inputs_embeds=output[i]).logits
                curr_log_prob = torch.log_softmax(logits_curr, dim=-1)
                kl = torch.nn.functional.kl_div(curr_log_prob, final_prob, reduction="batchmean")
                kl_values.append(kl)

        return kl_values

    def prob_final_pred_token(self, prompt):

        input_tkns = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model(**input_tkns, output_hidden_states=True).hidden_states
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

    def layer_disagreement(self, prompt, evolution_scale=10, candidate_premature_layers=[]):
        with torch.no_grad():
            input_tkns = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model(**input_tkns, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            premature_layers = []
            for h in hidden_states:
                logits = self.model.lm_head(h)
                premature_layers.append(logits)

            mature_logits = premature_layers[-1]
            seq_len = input_tkns.input_ids.shape[1]

            disagreement = []

            if candidate_premature_layers is None:
                num_layers = self.config.layers
                candidate_premature_layers = list(range(4, num_layers - 1, 4))

            # layer[batch, seq, vocab]
            for seq_i in range(seq_len - 1):
                token_results = {}

                stacked_premature = torch.stack(
                    [premature_layers[idx][:, seq_i, :] for idx in candidate_premature_layers]
                )

                stacked_premature = stacked_premature.squeeze(1)

                mature_token_logits = mature_logits[:, seq_i, :]

                # add mask
                mask = self.get_relative_top_filter(mature_token_logits, relative_top=0.1)

                # softmax
                softmax_mature = torch.nn.functional.softmax(mature_token_logits, dim=-1)
                softmax_premature = torch.nn.functional.softmax(stacked_premature, dim=-1)

                topk_prob, topk_indices = torch.topk(softmax_mature, evolution_scale, dim=-1)

                topk_indices = topk_indices[0]

                log_premature = torch.log_softmax(stacked_premature, dim=-1)
                log_mature = torch.log_softmax(mature_token_logits, dim=-1)
                divergence = log_premature - log_mature.unsqueeze(0)

                divergence = divergence.squeeze()

                candidate_gradients_expanded = softmax_premature.unsqueeze(1).expand(-1, len(topk_indices), -1).clone()

                layer_divergence_expanded = divergence.unsqueeze(1).expand(-1, evolution_scale, -1)
                layer_divergence_expanded = layer_divergence_expanded.to(self.device, dtype=torch.float32)

                topk_indices_expanded = topk_indices.unsqueeze(0).unsqueeze(2)

                candidate_mask = torch.zeros_like(candidate_gradients_expanded)
                candidate_mask.scatter_(
                    2,
                    topk_indices_expanded.expand(softmax_premature.size(0), -1, -1),
                    1,
                )
                candidate_gradients_expanded = candidate_gradients_expanded - candidate_mask

                candidate_gradients_expanded = candidate_gradients_expanded.to(self.device, dtype=torch.float32)

                # shape to vocab size
                vocab_size = candidate_gradients_expanded.shape[-1]
                flat_mask = mask.flatten()[:vocab_size]

                expanded_mask = flat_mask.view(1, 1, vocab_size).expand_as(candidate_gradients_expanded)

                # if rejected by mask cast aside
                candidate_gradients_expanded[expanded_mask] = 0.0
                layer_divergence_expanded[expanded_mask] = 0.0

                layer_dot_results = F.cosine_similarity(candidate_gradients_expanded, layer_divergence_expanded, dim=2)
                for i, layer_idx in enumerate(candidate_premature_layers):
                    token_results[layer_idx] = layer_dot_results[i].detach().cpu()
                disagreement.append(token_results)
            return disagreement

    def token_ranking_evolution(self, prompt):

        with torch.no_grad():
            input_tkns = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model(**input_tkns, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            premature_layers = []
            for h in hidden_states:
                logits = self.model.lm_head(h)
                premature_layers.append(logits)

            mature_logits = premature_layers[-1][:, -1, :]
            token_id = torch.argmax(mature_logits, dim=-1)
            ranking = []

            for layer in premature_layers:
                tkn_logits = layer[:, -1, :]  # get layer logits
                probs = torch.nn.functional.softmax(tkn_logits, dim=-1)

                sorted_idx = torch.argsort(probs, descending=True)  # sort
                rank = (sorted_idx == token_id.unsqueeze(-1)).nonzero(as_tuple=True)
                ranking.append(rank)
        return ranking


if __name__ == "__main__":
    model_name = "EleutherAI/gpt-neo-1.3B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_model = SLED_DecodedLLM_GSM8K(model_name, device)
    prompt = "You discover a new color that no human has ever seen. How would you describe it to someone?"

    layers_to_test = list(range(test_model.layers))

    disagreement_results = test_model.layer_disagreement(
        prompt, evolution_scale=5, candidate_premature_layers=layers_to_test
    )

    print(f"\n--- layer disagreement (SLED) results for {model_name} ---")

    tokens = test_model.tokenizer.tokenize(prompt)

    for i, token_data in enumerate(disagreement_results):
        token_text = tokens[i] if i < len(tokens) else f"Pos_{i}"
        print(f"\ntoken {i} ('{token_text}'):")

        for layer_idx, scores in token_data.items():
            avg_score = scores.mean().item()
            print(f"  layer {layer_idx:02d} | avg disagreement : {avg_score:.4f}")
