"""
niah_128k.py — Needle-In-A-Haystack benchmark for ECC KV cache.

Measures retrieval accuracy across context lengths and needle depths.
Runs all 3 methods (fp16, int4_bnb, int4_ecc) on IDENTICAL inputs per trial
to eliminate variance from random haystack construction.

Usage:
    python benchmarks/niah_128k.py \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --ctx-lengths 8000 32000 64000 128000 \\
        --depths 0.0 0.25 0.5 0.75 1.0 \\
        --trials 100 \\
        --output results/niah_results.json

Results format:
    {
      "config": {...},
      "results": {
        "fp16": {"8000": {"0.0": 0.99, ...}, ...},
        "int4_bnb": {...},
        "int4_ecc": {...}
      }
    }
"""
import argparse
import json
import random
import string
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─── Needle construction ───────────────────────────────────────────────────

TOPICS = [
    "starfish migration", "the lost city", "ancient recipe", "quantum key",
    "forgotten password", "secret ingredient", "hidden message", "code phrase",
    "emergency protocol", "backup sequence", "activation word", "cipher key",
    "treasure location", "safety override", "master passphrase",
]

def make_needle(topic: str, needle_id: str) -> str:
    return f"The special code for {topic} is {needle_id}."

def make_query(topic: str) -> str:
    return f"\n\nBased on the document above, what is the special code for {topic}? Answer with only the code."

def random_needle_id() -> str:
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))


# ─── Haystack management ───────────────────────────────────────────────────

def load_haystack_corpus() -> str:
    """Load haystack text (Paul Graham essays via HF datasets)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("pg19", split="train", streaming=True)
        corpus = ""
        for item in ds:
            corpus += item["text"] + "\n\n"
            if len(corpus) > 300_000:
                break
        return corpus
    except Exception:
        # Fallback: synthetic haystack
        sentences = [
            "The researchers found no significant correlation between the variables.",
            "Historical records indicate that trade routes shifted dramatically.",
            "The committee reviewed all submitted proposals and selected three finalists.",
            "Environmental factors play a crucial role in determining outcomes.",
        ] * 2000
        return " ".join(sentences)


def build_document(
    haystack: str,
    needle: str,
    depth: float,
    ctx_tokens: int,
    tokenizer,
) -> torch.Tensor:
    """
    Build a document of exactly ctx_tokens length with needle at depth position.

    depth=0.0 → needle at the very beginning
    depth=1.0 → needle at the very end
    """
    # Target haystack length (leave room for needle + query)
    needle_tokens = len(tokenizer.encode(needle))
    hay_token_budget = ctx_tokens - needle_tokens - 10  # 10 token buffer

    # Tokenize haystack
    hay_ids = tokenizer.encode(haystack, add_special_tokens=False)
    if len(hay_ids) > hay_token_budget:
        hay_ids = hay_ids[:hay_token_budget]

    # Insert needle at depth
    insert_pos = int(len(hay_ids) * depth)
    needle_ids = tokenizer.encode(needle, add_special_tokens=False)

    doc_ids = hay_ids[:insert_pos] + needle_ids + hay_ids[insert_pos:]
    doc_ids = doc_ids[:ctx_tokens]

    return doc_ids


# ─── Inference methods ─────────────────────────────────────────────────────

def infer_fp16(model, input_ids: torch.Tensor, tokenizer) -> str:
    """Standard FP16 inference."""
    with torch.no_grad():
        out = model.generate(
            input_ids.to("cuda").unsqueeze(0),
            max_new_tokens=10,
            do_sample=False,
            temperature=None,
            top_p=None,
            use_cache=True,
        )
    return tokenizer.decode(out[0, input_ids.shape[0]:], skip_special_tokens=True).strip()


def infer_int4_bnb(model_bnb, input_ids: torch.Tensor, tokenizer) -> str:
    """INT4 bitsandbytes quantized inference."""
    with torch.no_grad():
        out = model_bnb.generate(
            input_ids.to("cuda").unsqueeze(0),
            max_new_tokens=10,
            do_sample=False,
            temperature=None,
            top_p=None,
            use_cache=True,
        )
    return tokenizer.decode(out[0, input_ids.shape[0]:], skip_special_tokens=True).strip()


def infer_int4_ecc(model, input_ids: torch.Tensor, tokenizer, max_ctx: int) -> str:
    """ECC KV cache inference."""
    from custom_kv import ecc_cache

    with ecc_cache(model, batch_size=1, max_cache_len=max_ctx + 50) as cache:
        with torch.no_grad():
            out = model.generate(
                input_ids.to("cuda").unsqueeze(0),
                max_new_tokens=10,
                do_sample=False,
                temperature=None,
                top_p=None,
                past_key_values=cache,
                use_cache=True,
            )
    return tokenizer.decode(out[0, input_ids.shape[0]:], skip_special_tokens=True).strip()


def check_response(response: str, needle_id: str) -> bool:
    """Check if the needle ID appears in the response."""
    return needle_id.upper() in response.upper()


# ─── Main benchmark ────────────────────────────────────────────────────────

def run_niah_benchmark(
    model_id: str,
    ctx_lengths: List[int],
    depths: List[float],
    n_trials: int,
    methods: List[str],
    output_path: str,
    resume: bool = True,
) -> Dict:
    """
    Run the full NIAH benchmark grid.

    All methods run on identical (haystack, needle, depth) per trial.
    Results saved incrementally after each trial (crash-safe).
    """
    print(f"\n{'='*60}")
    print(f"NIAH Benchmark: {model_id}")
    print(f"Contexts: {ctx_lengths}")
    print(f"Depths: {depths}")
    print(f"Trials: {n_trials} | Methods: {methods}")
    print(f"{'='*60}\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load FP16 model
    print("Loading FP16 model...")
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda")
    model_fp16.eval()

    # Load INT4 bnb model (if needed)
    model_bnb = None
    if "int4_bnb" in methods:
        print("Loading INT4 bitsandbytes model...")
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True,
                                      bnb_4bit_compute_dtype=torch.float16)
        model_bnb = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_cfg, device_map="cuda")
        model_bnb.eval()

    # Load haystack corpus (once, reused across all trials)
    print("Loading haystack corpus...")
    haystack = load_haystack_corpus()
    topics = TOPICS * (n_trials // len(TOPICS) + 1)

    # Initialize or resume results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if resume and output_file.exists():
        print(f"Resuming from {output_path}")
        with open(output_path) as f:
            saved = json.load(f)
        results = saved.get("results", {m: {} for m in methods})
        completed = saved.get("completed_trials", {})
    else:
        results = {m: {} for m in methods}
        completed = {}  # {f"{ctx}_{depth}_{trial}": True}

    config = {
        "model_id": model_id,
        "ctx_lengths": ctx_lengths,
        "depths": depths,
        "n_trials": n_trials,
        "methods": methods,
    }

    # ── Main benchmark loop ───────────────────────────────────────────
    total_trials = len(ctx_lengths) * len(depths) * n_trials
    trial_count = 0
    start_time = time.time()

    for ctx_len in ctx_lengths:
        for depth in depths:
            cell_key = f"{ctx_len}_{depth}"

            # Initialize result storage for this cell
            for m in methods:
                if str(ctx_len) not in results[m]:
                    results[m][str(ctx_len)] = {}
                if str(depth) not in results[m][str(ctx_len)]:
                    results[m][str(ctx_len)][str(depth)] = []

            correct = {m: 0 for m in methods}
            trials_done = len(results["fp16"].get(str(ctx_len), {}).get(str(depth), []))

            for trial in range(trials_done, n_trials):
                trial_key = f"{cell_key}_{trial}"
                trial_count += 1

                # Set seed for reproducibility
                random.seed(trial * 1000 + int(depth * 100))

                # Build identical inputs for all methods
                topic = topics[trial % len(topics)]
                needle_id = random_needle_id()
                needle = make_needle(topic, needle_id)
                query = make_query(topic)

                doc_ids = build_document(haystack, needle, depth, ctx_len - 20, tokenizer)
                query_ids = tokenizer.encode(query, add_special_tokens=False)
                full_ids = torch.tensor(doc_ids + query_ids)

                # Truncate to ctx_len
                if len(full_ids) > ctx_len:
                    full_ids = full_ids[:ctx_len]

                # Run all methods
                responses = {}
                if "fp16" in methods:
                    responses["fp16"] = infer_fp16(model_fp16, full_ids, tokenizer)

                if "int4_bnb" in methods and model_bnb is not None:
                    responses["int4_bnb"] = infer_int4_bnb(model_bnb, full_ids, tokenizer)

                if "int4_ecc" in methods:
                    responses["int4_ecc"] = infer_int4_ecc(
                        model_fp16, full_ids, tokenizer, ctx_len)

                # Score all methods
                for m in methods:
                    hit = check_response(responses.get(m, ""), needle_id)
                    results[m][str(ctx_len)][str(depth)].append(int(hit))

                # Progress reporting
                elapsed = time.time() - start_time
                eta = elapsed / trial_count * (total_trials - trial_count)
                current_acc = {m: sum(results[m][str(ctx_len)][str(depth)]) /
                               len(results[m][str(ctx_len)][str(depth)])
                               for m in methods}

                print(
                    f"  ctx={ctx_len:6d} depth={depth:.2f} trial={trial+1:3d}/{n_trials} "
                    f"| {' '.join(f'{m}={current_acc[m]:.1%}' for m in methods)} "
                    f"| ETA: {eta/60:.1f}min"
                )

                # Save incrementally (every 5 trials)
                if trial % 5 == 4 or trial == n_trials - 1:
                    _save_results(output_path, config, results)

    # Final save with accuracy aggregation
    final = _aggregate_results(results, n_trials)
    output = {"config": config, "results": results, "accuracy": final}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")
    _print_summary(final, methods)
    return output


def _save_results(path, config, results):
    """Incremental save (crash-safe)."""
    tmp = Path(path).with_suffix(".tmp.json")
    with open(tmp, "w") as f:
        json.dump({"config": config, "results": results}, f)
    tmp.rename(path)


def _aggregate_results(results, n_trials):
    """Compute accuracy as mean of binary hit list."""
    agg = {}
    for method, ctx_data in results.items():
        agg[method] = {}
        for ctx, depth_data in ctx_data.items():
            agg[method][ctx] = {}
            for depth, hits in depth_data.items():
                if hits:
                    agg[method][ctx][depth] = sum(hits) / len(hits)
    return agg


def _print_summary(accuracy, methods):
    """Print summary table."""
    print(f"\n{'='*60}")
    print("NIAH ACCURACY SUMMARY")
    print(f"{'='*60}")
    for method in methods:
        if method not in accuracy:
            continue
        ctx_accs = []
        for ctx_data in accuracy[method].values():
            ctx_accs.extend(ctx_data.values())
        if ctx_accs:
            mean_acc = sum(ctx_accs) / len(ctx_accs)
            print(f"  {method:12s}: mean={mean_acc:.1%}")
    print(f"{'='*60}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NIAH Benchmark for ECC KV Cache")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--ctx-lengths", nargs="+", type=int,
                        default=[8000, 32000, 64000, 128000])
    parser.add_argument("--depths", nargs="+", type=float,
                        default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--methods", nargs="+", default=["fp16", "int4_bnb", "int4_ecc"])
    parser.add_argument("--output", default="results/niah_results.json")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    run_niah_benchmark(
        model_id=args.model,
        ctx_lengths=args.ctx_lengths,
        depths=args.depths,
        n_trials=args.trials,
        methods=args.methods,
        output_path=args.output,
        resume=not args.no_resume,
    )
