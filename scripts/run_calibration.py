"""
run_calibration.py — Run Lloyd-Max calibration on A100 and save config.

Usage:
    python scripts/run_calibration.py \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --n-samples 512 \\
        --output calibration_config.json
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from custom_kv.calibration import calibrate_from_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--output", default="calibration_config.json")
    parser.add_argument("--data-path", default="./data/calibration/")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda")
    model.eval()

    calibrate_from_model(
        model=model,
        tokenizer=tokenizer,
        calibration_data_path=args.data_path,
        n_samples=args.n_samples,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
