"""
A100 session master script — runs the complete 8-hour pipeline.

Run on A100 Colab:
    !python scripts/a100_session.py 2>&1 | tee session_log.txt

Phases (with checkpoints):
    0. Environment setup and CUDA extension build
    1. Calibration (512 samples)
    2. Smoke test at 2k context
    3. VRAM benchmark (8k, 32k, 64k, 128k)
    4. Throughput benchmark (64k context)
    5. NIAH 100-trial benchmark (main event)
    6. Generate comparison table + update README
    7. Git commit + tag + push

Estimated time: ~7.5 hours on A100 80GB
"""
import os
import sys
import json
import time
import subprocess
from pathlib import Path


MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ─── Checkpoint system ─────────────────────────────────────────────────────

CHECKPOINT_FILE = Path("results/.checkpoints.json")

def load_checkpoints():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}

def save_checkpoint(phase, status="done"):
    cps = load_checkpoints()
    cps[phase] = {"status": status, "time": time.strftime("%H:%M:%S")}
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(cps, f, indent=2)
    print(f"  ✓ Checkpoint saved: {phase}")

def is_done(phase):
    return load_checkpoints().get(phase, {}).get("status") == "done"


def run(cmd, **kwargs):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True, **kwargs)
    if result.returncode != 0:
        print(f"⚠ Command exited with code {result.returncode}")
    return result.returncode == 0


# ─── Phase 0: Environment ──────────────────────────────────────────────────

def phase_0_setup():
    if is_done("phase_0"):
        print("[Phase 0] Already done, skipping.")
        return

    print("\n" + "="*60)
    print("PHASE 0: Environment Setup")
    print("="*60)

    t0 = time.time()

    # Install dependencies
    run("pip install -q torch==2.3.0+cu121 transformers triton bitsandbytes "
        "scipy numpy datasets matplotlib gradio tqdm --extra-index-url "
        "https://download.pytorch.org/whl/cu121")

    # Build CUDA extension
    print("\nBuilding CUDA extension...")
    if not run("python setup.py build_ext --inplace"):
        print("❌ CUDA extension build FAILED. Check nvcc is available.")
        print("   Try: !apt-get install -y cuda-toolkit-11-8")
        sys.exit(1)

    # Verify extension loads
    try:
        import custom_ecc_cuda
        print("✓ custom_ecc_cuda loaded successfully")
    except ImportError as e:
        print(f"❌ Extension load failed: {e}")
        sys.exit(1)

    # Run CPU unit tests
    print("\nRunning CPU unit tests...")
    run("python -m pytest tests/unit/ -v -p no:dandi --tb=short")

    # Login to HF (token should be set in Colab secrets)
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        run(f"huggingface-cli login --token {hf_token}")
        print("✓ HF token configured")
    else:
        print("⚠ HF_TOKEN not set — model download may fail for gated models")

    elapsed = time.time() - t0
    print(f"\n[Phase 0 complete in {elapsed/60:.1f} min]")
    save_checkpoint("phase_0")


# ─── Phase 1: Calibration ─────────────────────────────────────────────────

def phase_1_calibration():
    if is_done("phase_1"):
        print("[Phase 1] Already done, skipping.")
        return

    print("\n" + "="*60)
    print("PHASE 1: Lloyd-Max Calibration")
    print("="*60)

    t0 = time.time()
    run(f"python scripts/run_calibration.py "
        f"--model {MODEL_ID} "
        f"--n-samples 512 "
        f"--output calibration_config.json")

    if not Path("calibration_config.json").exists():
        print("❌ Calibration failed — config not generated")
        sys.exit(1)

    with open("calibration_config.json") as f:
        config = json.load(f)

    print(f"  Calibrated {len(config['layers'])} layers")
    print(f"  Sample layer 0: {config['layers']['0']}")

    elapsed = time.time() - t0
    print(f"\n[Phase 1 complete in {elapsed/60:.1f} min]")
    save_checkpoint("phase_1")


# ─── Phase 2: Smoke test ──────────────────────────────────────────────────

def phase_2_smoke():
    if is_done("phase_2"):
        print("[Phase 2] Already done, skipping.")
        return

    print("\n" + "="*60)
    print("PHASE 2: Smoke Test")
    print("="*60)

    t0 = time.time()
    success = run(f"python scripts/smoke_test.py "
                  f"--model {MODEL_ID} "
                  f"--ctx-len 2000 "
                  f"--n-tokens 50")

    if not success:
        print("❌ SMOKE TEST FAILED — Do not proceed to benchmarks")
        print("   Debug steps:")
        print("   1. python -c \"import custom_ecc_cuda; print(dir(custom_ecc_cuda))\"")
        print("   2. Check for CUDA kernel errors in build log")
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"\n[Phase 2 complete in {elapsed/60:.1f} min]")
    save_checkpoint("phase_2")


# ─── Phase 3: VRAM benchmark ──────────────────────────────────────────────

def phase_3_vram():
    if is_done("phase_3"):
        print("[Phase 3] Already done, skipping.")
        return

    print("\n" + "="*60)
    print("PHASE 3: VRAM Reduction Benchmark")
    print("="*60)

    t0 = time.time()
    run(f"python benchmarks/vram_reduction.py "
        f"--model {MODEL_ID} "
        f"--ctx-lengths 8000 32000 64000 128000 "
        f"--output results/vram_results.json")

    elapsed = time.time() - t0
    print(f"\n[Phase 3 complete in {elapsed/60:.1f} min]")
    save_checkpoint("phase_3")


# ─── Phase 4: Throughput ──────────────────────────────────────────────────

def phase_4_throughput():
    if is_done("phase_4"):
        print("[Phase 4] Already done, skipping.")
        return

    print("\n" + "="*60)
    print("PHASE 4: Throughput Benchmark")
    print("="*60)

    t0 = time.time()
    run(f"python benchmarks/throughput.py "
        f"--model {MODEL_ID} "
        f"--ctx-len 64000 "
        f"--n-tokens 200 "
        f"--output results/throughput.json")

    elapsed = time.time() - t0
    print(f"\n[Phase 4 complete in {elapsed/60:.1f} min]")
    save_checkpoint("phase_4")


# ─── Phase 5: NIAH (the long one) ─────────────────────────────────────────

def phase_5_niah():
    if is_done("phase_5"):
        print("[Phase 5] Already done, skipping.")
        return

    print("\n" + "="*60)
    print("PHASE 5: NIAH 100-Trial Benchmark (~4.5 hours)")
    print("="*60)

    t0 = time.time()

    # Run with resume support — if interrupted, restart this script and
    # it will pick up from the last saved trial
    run(f"python benchmarks/niah_128k.py "
        f"--model {MODEL_ID} "
        f"--ctx-lengths 8000 32000 64000 128000 "
        f"--depths 0.0 0.25 0.5 0.75 1.0 "
        f"--trials 100 "
        f"--methods fp16 int4_bnb int4_ecc "
        f"--output results/niah_results.json")

    elapsed = time.time() - t0
    print(f"\n[Phase 5 complete in {elapsed/60:.1f} min]")
    save_checkpoint("phase_5")


# ─── Phase 6: Publish ─────────────────────────────────────────────────────

def phase_6_publish():
    print("\n" + "="*60)
    print("PHASE 6: Generate Comparison Table + Update README")
    print("="*60)

    # Generate comparison table
    run("python benchmarks/compare_baselines.py "
        "--vram results/vram_results.json "
        "--niah results/niah_results.json "
        "--throughput results/throughput.json "
        "--output results/comparison_table.md")

    print("\nManually copy comparison_table.md values into README.md")
    print("Then run: python scripts/update_readme.py")

    save_checkpoint("phase_6")


# ─── Phase 7: Git push ────────────────────────────────────────────────────

def phase_7_git():
    print("\n" + "="*60)
    print("PHASE 7: Git Commit + Push")
    print("="*60)

    run('git config user.email "you@example.com"')
    run('git config user.name "Your Name"')
    run("git add -A")
    run('git commit -m "feat: ECC KV cache v0.1.0 benchmark results"')
    run("git push origin main")
    run("git tag v0.1.0")
    run("git push --tags")

    print("\n✓ Repository published")
    save_checkpoint("phase_7")


# ─── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    total_start = time.time()

    print(f"\n{'='*60}")
    print(f"ECC KV Cache — A100 Session Master Script")
    print(f"Model: {MODEL_ID}")
    print(f"{'='*60}\n")

    phase_0_setup()
    phase_1_calibration()
    phase_2_smoke()
    phase_3_vram()
    phase_4_throughput()
    phase_5_niah()
    phase_6_publish()
    phase_7_git()

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"SESSION COMPLETE in {total_elapsed/3600:.2f} hours")
    print(f"{'='*60}")
