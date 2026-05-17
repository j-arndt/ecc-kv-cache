"""Fix _collect_kv_activations in calibration.py to use k_proj/v_proj hooks."""
import re

with open("custom_kv/calibration.py", encoding="utf-8") as f:
    content = f.read()

new_func = '''\
def _collect_kv_activations(
    model,
    tokenizer,
    texts,
    max_length=2048,
    device="cuda",
):
    """
    Run forward passes and collect KV activations per layer.
    Hooks on k_proj/v_proj directly -- avoids DynamicCache format
    changes in transformers >= 4.36 that broke the old attention hook.
    """
    import torch

    layer_kvs = {}
    num_layers = model.config.num_hidden_layers
    for i in range(num_layers):
        layer_kvs[i] = {"k": [], "v": []}

    hooks = []

    def make_k_hook(layer_idx):
        def hook(module, input, output):
            k = output.detach().float().reshape(-1, output.shape[-1])
            n = min(1000, k.shape[0])
            layer_kvs[layer_idx]["k"].append(k[torch.randperm(k.shape[0])[:n]].cpu())
        return hook

    def make_v_hook(layer_idx):
        def hook(module, input, output):
            v = output.detach().float().reshape(-1, output.shape[-1])
            n = min(1000, v.shape[0])
            layer_kvs[layer_idx]["v"].append(v[torch.randperm(v.shape[0])[:n]].cpu())
        return hook

    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.k_proj.register_forward_hook(make_k_hook(i)))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(make_v_hook(i)))

    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text, return_tensors="pt",
                max_length=max_length, truncation=True,
            ).to(device)
            model(**inputs)

    for h in hooks:
        h.remove()

    result = {}
    for i in range(num_layers):
        if layer_kvs[i]["k"]:
            result[i] = {
                "k": torch.cat(layer_kvs[i]["k"], dim=0),
                "v": torch.cat(layer_kvs[i]["v"], dim=0),
            }
    return result
'''

# Replace old function (from def to end of return result block)
start = content.index("def _collect_kv_activations(")
end = content.index("\n    return result\n", start) + len("\n    return result\n")
new_content = content[:start] + new_func + "\n" + content[end:]

with open("custom_kv/calibration.py", "w", encoding="utf-8") as f:
    f.write(new_content)

print(f"Done. {new_content.count(chr(10))} lines written.")
