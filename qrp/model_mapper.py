
import torch

def get_layer_structure(layer):
    if hasattr(layer, "self_attn") and hasattr(layer, "mlp"):
        return (layer.self_attn, ["q_proj", "k_proj", "v_proj", "o_proj"]), \
               (layer.mlp, ["gate_proj", "up_proj", "down_proj"])

    if hasattr(layer, "conv") and hasattr(layer, "feed_forward"):
        return (layer.conv, ["in_proj", "out_proj"]), \
               (layer.feed_forward, ["w1", "w2", "w3"])

    if hasattr(layer, "shared_mlp"):
        attn_part = None
        attn_projs = []
        if getattr(layer, "self_attn", None) is not None:
            attn_part = layer.self_attn
            attn_projs = ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif getattr(layer, "mamba", None) is not None:
            attn_part = layer.mamba
            attn_projs = ["in_proj", "out_proj"]
        return (attn_part, attn_projs), \
               (layer.shared_mlp, ["input_linear", "output_linear"])

    if not hasattr(layer, "self_attn") or not hasattr(layer, "mlp"):
        print(f"Warning: Unknown layer structure for {type(layer)}. Falling back to Llama defaults.")

    return (getattr(layer, "self_attn", None), ["q_proj", "k_proj", "v_proj", "o_proj"]), \
           (getattr(layer, "mlp", None), ["gate_proj", "up_proj", "down_proj"])

def get_model_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    elif hasattr(model, "layers"):
        return model.layers
    else:
        raise AttributeError("Model does not have model.layers or layers attribute.")

