"""
vram_reduction.py — VRAM consumption benchmark comparing FP16 vs INT4 ECC cache.

Measures peak GPU memory during prefill at various context lengths.

Usage:
    python benchmarks/vram_reduction.py \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --ctx-lengths 8000 32000 64000 128000 \\
        --output results/vram_results.json
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def measure_vram(
    model_id: str,
    ctx_lengths: list,
    output_path: str,
):
    print(f"\nVRAM Benchmark: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda")
    model.eval()

    # Baseline: model weights alone
    torch.cuda.synchronize()
    weight_vram = torch.cuda.memory_allocated() / 1e9
    print(f"Model weights: {weight_vram:.2f} GB")

    results = {}
    # Simple haystack text
    haystack_word = "The researchers concluded that the data was inconclusive. "

    for ctx_len in ctx_lengths:
        print(f"\n  Context length: {ctx_len:,} tokens")
        text = haystack_word * (ctx_len // 8 + 1)
        inputs = tokenizer(text, return_tensors="pt",
                           max_length=ctx_len, truncation=True).to("cuda")
        actual_len = inputs["input_ids"].shape[1]
        print(f"    Actual token count: {actual_len:,}")

        # ── FP16 baseline ────────────────────────────────────────────
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        try:
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=1, use_cache=True)
            torch.cuda.synchronize()
            fp16_peak = torch.cuda.max_memory_allocated() / 1e9
        except torch.cuda.OutOfMemoryError:
            fp16_peak = None
            print(f"    FP16: OOM at {ctx_len} tokens")
        torch.cuda.empty_cache()

        # ── ECC KV cache ─────────────────────────────────────────────
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        try:
            from custom_kv import ecc_cache
            with ecc_cache(model, batch_size=1, max_cache_len=actual_len + 10) as cache:
                with torch.no_grad():
                    model.generate(**inputs, max_new_tokens=1,
                                   use_cache=True, past_key_values=cache)
            torch.cuda.synchronize()
            ecc_peak = torch.cuda.max_memory_allocated() / 1e9
        except torch.cuda.OutOfMemoryError:
            ecc_peak = None
            print(f"    ECC:  OOM at {ctx_len} tokens")
        torch.cuda.empty_cache()

        # Compute ratio
        if fp16_peak and ecc_peak:
            ratio = round(fp16_peak / ecc_peak, 2)
        else:
            ratio = None

        results[ctx_len] = {
            "ctx_len": ctx_len,
            "actual_tokens": actual_len,
            "fp16_peak_gb": round(fp16_peak, 2) if fp16_peak else None,
            "ecc_peak_gb":  round(ecc_peak, 2)  if ecc_peak  else None,
            "reduction_x":  ratio,
            "fp16_kv_only_gb": round(fp16_peak - weight_vram, 2) if fp16_peak else None,
            "ecc_kv_only_gb":  round(ecc_peak  - weight_vram, 2) if ecc_peak  else None,
        }

        print(f"    FP16: {fp16_peak:.2f} GB" if fp16_peak else "    FP16: OOM")
        print(f"    ECC:  {ecc_peak:.2f} GB"  if ecc_peak  else "    ECC:  OOM")
        print(f"    Ratio: {ratio}x"           if ratio     else "    Ratio: N/A")

    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({
            "model_id": model_id,
            "weight_vram_gb": round(weight_vram, 2),
            "results": results,
        }, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")

    # Summary table
    print(f"\n{'Context':>10} {'FP16':>10} {'ECC':>10} {'Ratio':>8}")
    print("-" * 42)
    for ctx, r in results.items():
        fp16 = f"{r['fp16_peak_gb']:.2f}GB" if r['fp16_peak_gb'] else "OOM"
        ecc  = f"{r['ecc_peak_gb']:.2f}GB"  if r['ecc_peak_gb']  else "OOM"
        ratio = f"{r['reduction_x']}x"       if r['reduction_x']  else "N/A"
        print(f"{ctx:>10,} {fp16:>10} {ecc:>10} {ratio:>8}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--ctx-lengths", nargs="+", type=int,
                        default=[8000, 32000, 64000, 128000])
    parser.add_argument("--output", default="results/vram_results.json")
    args = parser.parse_args()
    measure_vram(args.model, args.ctx_lengths, args.output)
