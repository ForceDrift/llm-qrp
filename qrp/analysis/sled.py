import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

class SLED_Decoded:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.model, self.tokenizer = self.load_model(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.layers = self.config.num_hidden_layers
        self.lm_head = self.model.get_output_embeddings()
        
    def load_model(self, model_name):
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.to(self.device)
        return model, tokenizer

    def compute_layer_logits_to_cpu(self, prompt):
        """Single forward pass. Returns logits for every layer on CPU to save GPU memory."""
        input_tkns = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**input_tkns, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            all_logits_cpu = []
            for h in hidden_states:
                logits = self.lm_head(h).float().cpu()
                all_logits_cpu.append(logits)
            del outputs, hidden_states
            torch.cuda.empty_cache()
        return all_logits_cpu, input_tkns.input_ids

    def kl_between_current_prev_from_cache(self, all_logits_cpu):
        """KL divergence between consecutive layers, using CPU logits."""
        kl_values = []
        for i in range(1, len(all_logits_cpu)):
            logits_prev = all_logits_cpu[i - 1]
            logits_curr = all_logits_cpu[i]
            prev_prob = torch.softmax(logits_prev, dim=-1)
            curr_log_prob = torch.log_softmax(logits_curr, dim=-1)
            kl = torch.nn.functional.kl_div(curr_log_prob, prev_prob, reduction="batchmean")
            kl_values.append(kl)
        return kl_values

    def kl_between_current_prev(self, prompt):
        input_tkns = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**input_tkns, output_hidden_states=True).hidden_states
            kl_values = []

            for i in range(1, len(output)):
                logits_prev = self.lm_head(output[i - 1]).float()
                logits_curr = self.lm_head(output[i]).float()

                prev_prob = torch.softmax(logits_prev, dim=-1)
                curr_log_prob = torch.log_softmax(logits_curr, dim=-1)
                kl = torch.nn.functional.kl_div(curr_log_prob, prev_prob, reduction="batchmean")
                print(f"layer {i} and layer {i - 1}, kl-div: {kl}")
                kl_values.append(kl)

        return kl_values

    def kl_between_current_final(self, prompt):

        input_tkns = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**input_tkns, output_hidden_states=True)

            if isinstance(outputs, tuple):
                logits_final = outputs[0]
                hidden_states = outputs.hidden_states if hasattr(outputs, "hidden_states") else outputs[1] # type: ignore
            else:   
                logits_final = outputs.logits  
                hidden_states = outputs.hidden_states 

            final_prob = torch.log_softmax(logits_final, dim=-1)

            kl_values = []

            for i in range(1, len(outputs)):
                logits_curr = self.model.lm_head(hidden_states[i])
                curr_log_prob = torch.log_softmax(logits_curr, dim=-1)
                kl = torch.nn.functional.kl_div(curr_log_prob, final_prob, reduction="batchmean")
                kl_values.append(kl)

        return kl_values

    def prob_final_pred_token(self, prompt):

        input_tkns = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**input_tkns, output_hidden_states=True)
            logits = outputs.logits
        
        last_token_logits = logits[:,-1,:]


        prob = torch.nn.functional.softmax(last_token_logits, dim=-1)
        tkn_prob, token_id = torch.max(prob, dim=-1)
        pred_token = self.tokenizer.convert_ids_to_tokens([token_id[0].item()])[0]
        return pred_token, tkn_prob[0].item()

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
        return scores_prob < probs_thresh

    def layer_disagreement_from_cache(self, all_logits_cpu, input_ids, evolution_scale=10, candidate_premature_layers=None):
        """Layer disagreement using CPU logits. Moves tensors to GPU only temporarily."""
        num_layers_total = len(all_logits_cpu)

        if candidate_premature_layers is None:
            candidate_premature_layers = list(range(4, num_layers_total - 1, 4))

        mature_logits_cpu = all_logits_cpu[num_layers_total - 1]
        seq_len = input_ids.shape[1]

        disagreement = []

        for seq_i in range(seq_len - 1):
            token_results = {}

            # Build stacked premature logits on CPU, then move to GPU for computation
            premature_stack = torch.stack(
                [all_logits_cpu[idx][:, seq_i, :] for idx in candidate_premature_layers]
            ).squeeze(1)

            mature_token_logits = mature_logits_cpu[:, seq_i, :]

            # Move to GPU for computation
            mature_gpu = mature_token_logits.to(self.device)
            premature_gpu = premature_stack.to(self.device)

            mask = self.get_relative_top_filter(mature_gpu, relative_top=0.1)

            softmax_mature = torch.nn.functional.softmax(mature_gpu, dim=-1)
            softmax_premature = torch.nn.functional.softmax(premature_gpu, dim=-1)

            topk_prob, topk_indices = torch.topk(softmax_mature, evolution_scale, dim=-1)
            topk_indices = topk_indices[0]

            log_premature = torch.log_softmax(premature_gpu.float(), dim=-1)
            log_mature = torch.log_softmax(mature_gpu.float(), dim=-1)
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

            vocab_size = candidate_gradients_expanded.shape[-1]
            flat_mask = mask.flatten()[:vocab_size]
            expanded_mask = flat_mask.view(1, 1, vocab_size).expand_as(candidate_gradients_expanded)
            candidate_gradients_expanded[expanded_mask] = 0.0
            layer_divergence_expanded[expanded_mask] = 0.0

            layer_dot_results = F.cosine_similarity(candidate_gradients_expanded, layer_divergence_expanded, dim=2)
            for i, layer_idx in enumerate(candidate_premature_layers):
                token_results[layer_idx] = layer_dot_results[i].cpu()
            disagreement.append(token_results)

            # Free GPU tensors immediately
            del mature_gpu, premature_gpu, softmax_mature, softmax_premature
            del log_premature, log_mature, divergence
            del candidate_gradients_expanded, layer_divergence_expanded
            del candidate_mask, expanded_mask, layer_dot_results, topk_indices_expanded
            torch.cuda.empty_cache()

        return disagreement

    def layer_disagreement(self, prompt, evolution_scale=10, candidate_premature_layers=[]):
        with torch.no_grad():
            input_tkns = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model(**input_tkns, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            num_layers_total = len(hidden_states)

            if candidate_premature_layers is None:
                candidate_premature_layers = list(range(4, num_layers_total - 1, 4))

            needed_indices = set(candidate_premature_layers) | {num_layers_total - 1}

            premature_layers = {}
            for idx in needed_indices:
                premature_layers[idx] = self.lm_head(hidden_states[idx])
            del outputs, hidden_states

            mature_logits = premature_layers[num_layers_total - 1]
            seq_len = input_tkns.input_ids.shape[1]

            disagreement = []

            for seq_i in range(seq_len - 1):
                token_results = {}

                stacked_premature = torch.stack(
                    [premature_layers[idx][:, seq_i, :] for idx in candidate_premature_layers]
                )

                stacked_premature = stacked_premature.squeeze(1)

                mature_token_logits = mature_logits[:, seq_i, :]

                mask = self.get_relative_top_filter(mature_token_logits, relative_top=0.1)

                softmax_mature = torch.nn.functional.softmax(mature_token_logits, dim=-1)
                softmax_premature = torch.nn.functional.softmax(stacked_premature, dim=-1)

                topk_prob, topk_indices = torch.topk(softmax_mature, evolution_scale, dim=-1)

                topk_indices = topk_indices[0]

                log_premature = torch.log_softmax(stacked_premature.float(), dim=-1)
                log_mature = torch.log_softmax(mature_token_logits.float(), dim=-1)
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

                vocab_size = candidate_gradients_expanded.shape[-1]
                flat_mask = mask.flatten()[:vocab_size]

                expanded_mask = flat_mask.view(1, 1, vocab_size).expand_as(candidate_gradients_expanded)

                candidate_gradients_expanded[expanded_mask] = 0.0
                layer_divergence_expanded[expanded_mask] = 0.0

                layer_dot_results = F.cosine_similarity(candidate_gradients_expanded, layer_divergence_expanded, dim=2)
                for i, layer_idx in enumerate(candidate_premature_layers):
                    token_results[layer_idx] = layer_dot_results[i].detach().cpu()
                disagreement.append(token_results)

                del stacked_premature, softmax_mature, softmax_premature
                del log_premature, log_mature, divergence
                del candidate_gradients_expanded, layer_divergence_expanded
                del candidate_mask, expanded_mask, layer_dot_results

            del premature_layers, mature_logits
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

            for layer_idx, layer in enumerate(premature_layers):
                tkn_logits = layer[:, -1, :]
                probs = torch.nn.functional.softmax(tkn_logits, dim=-1)
                

                sorted_idx = torch.argsort(probs, descending=True)
                ranking.append({layer_idx: sorted_idx})
        return ranking
    @torch.no_grad()

    def sled_step(self, input_ids, evolution_scale=10, evolution_rate=2, candidate_layers=None):
        """
        Perform a single SLED decoding step.
        Uses disagreement signals to re-rank tokens.
        """
        if candidate_layers is None:
            candidate_layers = list(range(4, self.layers - 1, 4))

        outputs = self.model(input_ids, output_hidden_states=True)
        mature_logits = self.model.lm_head(outputs.hidden_states[-1][:, -1, :]) # [1, vocab]
        
        premature_layers = []
        for h in outputs.hidden_states:
            logits = self.model.lm_head(h[:, -1, :])
            premature_layers.append(logits)

        stacked_premature = torch.stack([premature_layers[idx] for idx in candidate_layers]).squeeze(1) # [num_cand, vocab]
        # softmax
        softmax_mature = F.softmax(mature_logits, dim=-1)
        softmax_premature = F.softmax(stacked_premature, dim=-1)

        # divergence
        log_premature = F.log_softmax(stacked_premature, dim=-1)
        log_mature = F.log_softmax(mature_logits, dim=-1)
        divergence = log_premature - log_mature.unsqueeze(0)
        adjustment = torch.zeros_like(mature_logits)

        for i in range(len(candidate_layers)):
            adjustment += (log_mature - log_premature[i])
            
        return mature_logits + (evolution_rate * (adjustment / len(candidate_layers)))

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=128, evolution_scale=10, evolution_rate=2, candidate_layers=None):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        for _ in range(max_new_tokens):
            logits = self.sled_step(input_ids, evolution_scale, evolution_rate, candidate_layers)
            
            next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
                
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
