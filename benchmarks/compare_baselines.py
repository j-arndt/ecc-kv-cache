"""
compare_baselines.py — Generate comparison table from benchmark JSON results.

Reads vram_results.json, niah_results.json, throughput.json and outputs
a formatted Markdown table for the README and article.

Usage:
    python benchmarks/compare_baselines.py \\
        --vram results/vram_results.json \\
        --niah results/niah_results.json \\
        --throughput results/throughput.json \\
        --output results/comparison_table.md
"""
import argparse
import json
from pathlib import Path


def load_json(path):
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


def format_vram_table(vram_data):
    if not vram_data:
        return "VRAM data not available.\n"

    lines = [
        "### VRAM Reduction",
        "",
        "| Context | FP16 KV | ECC KV | Reduction |",
        "|:-------:|:-------:|:------:|:---------:|",
    ]

    for ctx, r in vram_data.get("results", {}).items():
        fp16 = f"{r['fp16_peak_gb']:.2f} GB" if r.get('fp16_peak_gb') else "OOM"
        ecc  = f"{r['ecc_peak_gb']:.2f} GB"  if r.get('ecc_peak_gb')  else "OOM"
        ratio = f"**{r['reduction_x']:.2f}x**" if r.get('reduction_x') else "N/A"
        lines.append(f"| {int(ctx):,} | {fp16} | {ecc} | {ratio} |")

    return "\n".join(lines) + "\n"


def format_niah_table(niah_data):
    if not niah_data:
        return "NIAH data not available.\n"

    accuracy = niah_data.get("accuracy", {})
    if not accuracy:
        return "NIAH results not yet aggregated.\n"

    methods = list(accuracy.keys())
    ctx_lengths = list(next(iter(accuracy.values())).keys()) if accuracy else []

    lines = [
        "### Needle-In-A-Haystack Accuracy",
        "",
        f"| Method | {' | '.join(f'{int(c):,} tok' for c in ctx_lengths)} | Mean |",
        f"|:------:|{':-----:|' * (len(ctx_lengths) + 1)}",
    ]

    for method in methods:
        accs = []
        for ctx in ctx_lengths:
            ctx_data = accuracy.get(method, {}).get(ctx, {})
            if ctx_data:
                mean_acc = sum(ctx_data.values()) / len(ctx_data)
                accs.append(f"{mean_acc:.1%}")
            else:
                accs.append("N/A")

        all_accs = [float(a.rstrip('%')) / 100 for a in accs if a != "N/A"]
        overall_mean = f"{sum(all_accs)/len(all_accs):.1%}" if all_accs else "N/A"

        bold_open = "**" if method == "int4_ecc" else ""
        bold_close = "**" if method == "int4_ecc" else ""
        lines.append(
            f"| {bold_open}{method}{bold_close} | "
            f"{' | '.join(accs)} | {bold_open}{overall_mean}{bold_close} |"
        )

    return "\n".join(lines) + "\n"


def format_throughput_table(tput_data):
    if not tput_data:
        return "Throughput data not available.\n"

    results = tput_data.get("results", {})
    lines = [
        "### Decode Throughput",
        f"(Context: {tput_data.get('ctx_len', 'N/A'):,} tokens, "
        f"Decode: {tput_data.get('n_decode_tokens', 'N/A')} tokens)",
        "",
        "| Method | tok/s | vs FP16 |",
        "|:------:|:-----:|:-------:|",
    ]

    for method, r in results.items():
        tps = f"{r.get('tps', 'N/A'):.0f}" if isinstance(r.get('tps'), float) else "N/A"
        speedup = f"{r.get('speedup_vs_fp16', 'N/A'):.2f}x" if r.get('speedup_vs_fp16') else "baseline"
        bold = method == "int4_ecc"
        prefix = "**" if bold else ""
        suffix = "**" if bold else ""
        lines.append(f"| {prefix}{method}{suffix} | {prefix}{tps}{suffix} | {speedup} |")

    return "\n".join(lines) + "\n"


def generate_comparison_table(vram_path, niah_path, tput_path, output_path):
    vram_data = load_json(vram_path)
    niah_data = load_json(niah_path)
    tput_data = load_json(tput_path)

    sections = [
        "# ECC KV Cache — Benchmark Results\n",
        format_vram_table(vram_data),
        "",
        format_niah_table(niah_data),
        "",
        format_throughput_table(tput_data),
    ]

    table_md = "\n".join(sections)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(table_md)
        print(f"✓ Comparison table written to {output_path}")

    print(table_md)
    return table_md


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vram", default="results/vram_results.json")
    parser.add_argument("--niah", default="results/niah_results.json")
    parser.add_argument("--throughput", default="results/throughput.json")
    parser.add_argument("--output", default="results/comparison_table.md")
    args = parser.parse_args()

    generate_comparison_table(args.vram, args.niah, args.throughput, args.output)
