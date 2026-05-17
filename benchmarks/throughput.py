"""
throughput.py — Tokens-per-second benchmark: FP16 vs INT4 ECC vs INT4 bnb.

Measures sustained decode throughput at a fixed context length.

Usage:
    python benchmarks/throughput.py \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --ctx-len 64000 \\
        --n-tokens 200 \\
        --output results/throughput.json
"""
import argparse
import json
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def bench_throughput(model_id, ctx_len, n_tokens, output_path, warmup=2):
    print(f"\nThroughput Benchmark @ ctx={ctx_len:,} tokens, decode={n_tokens} tokens")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda")
    model.eval()

    # Build fixed prompt
    text = "The history of artificial intelligence spans many decades. " * (ctx_len // 10)
    inputs = tokenizer(text, return_tensors="pt",
                       max_length=ctx_len, truncation=True).to("cuda")

    results = {}

    def measure(label, generate_fn):
        # Warmup
        for _ in range(warmup):
            generate_fn()
            torch.cuda.synchronize()

        # Timed run
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        generate_fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tps = n_tokens / elapsed
        print(f"  {label:12s}: {tps:.0f} tok/s ({elapsed:.2f}s for {n_tokens} tokens)")
        results[label] = {"tps": round(tps, 1), "elapsed_s": round(elapsed, 3)}

    # ── FP16 baseline ─────────────────────────────────────────────────
    def fp16_gen():
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=n_tokens,
                           do_sample=False, use_cache=True)
    measure("fp16", fp16_gen)

    # ── ECC cache ─────────────────────────────────────────────────────
    from custom_kv import ecc_cache

    def ecc_gen():
        with ecc_cache(model, batch_size=1, max_cache_len=ctx_len + n_tokens + 10) as cache:
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=n_tokens,
                               do_sample=False, use_cache=True, past_key_values=cache)
    measure("int4_ecc", ecc_gen)

    # ── INT4 bitsandbytes ─────────────────────────────────────────────
    try:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True,
                                      bnb_4bit_compute_dtype=torch.float16)
        model_bnb = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_cfg, device_map="cuda")
        model_bnb.eval()

        def bnb_gen():
            with torch.no_grad():
                model_bnb.generate(**inputs, max_new_tokens=n_tokens,
                                   do_sample=False, use_cache=True)
        measure("int4_bnb", bnb_gen)
    except Exception as e:
        print(f"  int4_bnb: failed ({e})")

    # Compute speedups
    if "fp16" in results:
        for method in ["int4_ecc", "int4_bnb"]:
            if method in results:
                results[method]["speedup_vs_fp16"] = round(
                    results[method]["tps"] / results["fp16"]["tps"], 2)

    output = {
        "model_id": model_id,
        "ctx_len": ctx_len,
        "n_decode_tokens": n_tokens,
        "results": results,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")

    if "fp16" in results and "int4_ecc" in results:
        speedup = results["int4_ecc"]["speedup_vs_fp16"]
        print(f"\nECC speedup vs FP16: {speedup}x")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--ctx-len", type=int, default=64000)
    parser.add_argument("--n-tokens", type=int, default=200)
    parser.add_argument("--output", default="results/throughput.json")
    args = parser.parse_args()
    bench_throughput(args.model, args.ctx_len, args.n_tokens, args.output)
