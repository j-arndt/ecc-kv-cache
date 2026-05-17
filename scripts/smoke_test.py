"""
smoke_test.py — Quick end-to-end sanity check before full benchmarks.

Run on A100 immediately after building CUDA extension:
    python scripts/smoke_test.py --ctx-len 2000 --n-tokens 50

Expected: coherent text output, no errors, VRAM usage reported.
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def smoke_test(model_id: str, ctx_len: int, n_tokens: int):
    print(f"\n{'='*50}")
    print(f"ECC KV Cache Smoke Test")
    print(f"Model: {model_id}")
    print(f"Context: {ctx_len} tokens | Generate: {n_tokens} tokens")
    print(f"{'='*50}\n")

    # Load model
    print("[1/4] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda")
    model.eval()
    print(f"  Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Build test prompt
    prompt = ("The history of computing is a story of continuous innovation. "
              "From early mechanical calculators to modern neural networks, "
              "the field has evolved dramatically. ") * (ctx_len // 40)

    inputs = tokenizer(prompt, return_tensors="pt",
                       max_length=ctx_len, truncation=True).to("cuda")
    actual_len = inputs["input_ids"].shape[1]
    print(f"  Prompt tokenized: {actual_len} tokens\n")

    # ── Test 1: FP16 baseline ─────────────────────────────────────────
    print("[2/4] FP16 baseline generation...")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out_fp16 = model.generate(**inputs, max_new_tokens=n_tokens,
                                  do_sample=False, use_cache=True)
    fp16_vram = torch.cuda.max_memory_allocated() / 1e9
    fp16_text = tokenizer.decode(out_fp16[0, actual_len:], skip_special_tokens=True)
    print(f"  Peak VRAM: {fp16_vram:.2f} GB")
    print(f"  Output: {fp16_text[:100]}...\n")

    # ── Test 2: ECC cache ─────────────────────────────────────────────
    print("[3/4] ECC cache generation...")
    from custom_kv import ecc_cache, ErrorCorrectedCache

    torch.cuda.reset_peak_memory_stats()
    with ecc_cache(model, batch_size=1, max_cache_len=actual_len + n_tokens + 10) as cache:
        print(f"  Cache allocated: {cache}")
        with torch.no_grad():
            out_ecc = model.generate(**inputs, max_new_tokens=n_tokens,
                                     do_sample=False, use_cache=True,
                                     past_key_values=cache)
    ecc_vram = torch.cuda.max_memory_allocated() / 1e9
    ecc_text = tokenizer.decode(out_ecc[0, actual_len:], skip_special_tokens=True)
    print(f"  Peak VRAM: {ecc_vram:.2f} GB")
    print(f"  Output: {ecc_text[:100]}...\n")

    # ── Test 3: Output comparison ─────────────────────────────────────
    print("[4/4] Comparing outputs...")
    fp16_words = set(fp16_text.lower().split())
    ecc_words  = set(ecc_text.lower().split())
    if fp16_words and ecc_words:
        overlap = len(fp16_words & ecc_words) / len(fp16_words | ecc_words)
    else:
        overlap = 0.0

    vram_reduction = fp16_vram / ecc_vram if ecc_vram > 0 else 0

    print(f"\n{'='*50}")
    print(f"SMOKE TEST RESULTS")
    print(f"{'='*50}")
    print(f"  FP16 VRAM:         {fp16_vram:.2f} GB")
    print(f"  ECC VRAM:          {ecc_vram:.2f} GB")
    print(f"  VRAM reduction:    {vram_reduction:.2f}x")
    print(f"  Output overlap:    {overlap:.1%}")
    print(f"{'='*50}")

    if vram_reduction >= 1.5:
        print("✓ VRAM reduction verified")
    else:
        print("⚠ VRAM reduction below expected — check kernel")

    if overlap >= 0.3:
        print("✓ Output coherence verified")
    else:
        print("⚠ Low output overlap — verify ECC correction")

    print("\n✓ Smoke test complete. Proceed to full benchmarks.\n")
    return {"fp16_vram": fp16_vram, "ecc_vram": ecc_vram,
            "vram_reduction": vram_reduction, "output_overlap": overlap}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--ctx-len", type=int, default=2000)
    parser.add_argument("--n-tokens", type=int, default=50)
    args = parser.parse_args()
    smoke_test(args.model, args.ctx_len, args.n_tokens)
