"""Generate the definitive Colab notebook for the ECC KV cache A100 session."""
import json

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

cells = []

def code(source_lines, cell_id):
    return {
        "cell_type": "code",
        "metadata": {"id": cell_id},
        "source": source_lines,
        "execution_count": None,
        "outputs": []
    }

def md(source_lines, cell_id):
    return {
        "cell_type": "markdown",
        "metadata": {"id": cell_id},
        "source": source_lines
    }

# ── Title ─────────────────────────────────────────────────────────────────
cells.append(md([
    "# ECC KV Cache — A100 80GB Session\n",
    "**Run cells top to bottom. On any restart, re-run Cell 1 only.**"
], "title"))

# ── CELL 1: Setup ─────────────────────────────────────────────────────────
cells.append(code([
    "# CELL 1 — Run every restart\n",
    "import os, sys, torch, warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# 1. Clone or pull repo\n",
    "if not os.path.exists('/content/ecc-kv-cache'):\n",
    "    os.system('git clone https://github.com/j-arndt/ecc-kv-cache /content/ecc-kv-cache')\n",
    "else:\n",
    "    os.system('git -C /content/ecc-kv-cache pull origin main')\n",
    "\n",
    "os.chdir('/content/ecc-kv-cache')\n",
    "os.makedirs('results', exist_ok=True)\n",
    "\n",
    "# Print git hash so we know exactly what code is running\n",
    "import subprocess\n",
    "git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()\n",
    "print(f'Repo at commit: {git_hash}')\n",
    "\n",
    "# 2. Install bitsandbytes (required for INT4-BNB baseline in NIAH)\n",
    "os.system('pip install -q bitsandbytes>=0.46.1')\n",
    "\n",
    "# 3. Tell system linker where torch libs are (fixes libc10.so not found)\n",
    "torch_lib = torch.__path__[0] + '/lib'\n",
    "os.system(f'echo {torch_lib} > /etc/ld.so.conf.d/torch.conf && ldconfig')\n",
    "\n",
    "# 4. Clean stale build artifacts (avoids ABI mismatch across torch versions)\n",
    "os.system('rm -rf build/ custom_ecc_cuda*.so')\n",
    "os.system('python setup.py build_ext --inplace 2>&1 | grep -E \"building|copying|error\"')\n",
    "\n",
    "# 5. Make custom_kv importable by all !python subprocesses\n",
    "sys.path.insert(0, '/content/ecc-kv-cache')\n",
    "os.environ['PYTHONPATH'] = '/content/ecc-kv-cache'\n",
    "\n",
    "# 6. Verify extension loads\n",
    "import custom_ecc_cuda\n",
    "print('CUDA extension:', list(filter(lambda x: not x.startswith('_'), dir(custom_ecc_cuda))))\n",
    "print('GPU:', torch.cuda.get_device_name(0))\n",
    "print('VRAM total:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')\n",
    "print('\\n✓ Cell 1 complete — proceed to Cell 2')\n",
], "cell1"))

# ── CELL 2: Tests ─────────────────────────────────────────────────────────
# Run ALL tests including integration tests (catches Colab-specific regressions)
cells.append(code([
    "# CELL 2 — Full test suite including Colab-compat integration tests\n",
    "!python -m pytest tests/ -v --tb=short\n",
], "cell2"))

# ── CELL 3: HF Login ──────────────────────────────────────────────────────
cells.append(code([
    "# CELL 3 — HuggingFace login (once per VM session)\n",
    "from google.colab import userdata\n",
    "from huggingface_hub import login\n",
    "import os\n",
    "\n",
    "token = userdata.get('hf_token2')  # matches your Colab secret name\n",
    "os.environ['HF_TOKEN'] = token\n",
    "login(token=token, add_to_git_credential=False)\n",
    "print('\\n✓ Logged in')\n",
], "cell3"))

# ── CELL 4: Calibration ───────────────────────────────────────────────────
cells.append(code([
    "# CELL 4 — Calibration (~10 min). Skips if already done this session.\n",
    "import os\n",
    "\n",
    "if os.path.exists('calibration_config.json'):\n",
    "    import json\n",
    "    cfg = json.load(open('calibration_config.json'))\n",
    "    n = len(cfg.get('layers', {}))\n",
    "    print(f'Calibration already done: {n} layers. Skipping.')\n",
    "else:\n",
    "    os.system('git -C /content/ecc-kv-cache pull origin main')\n",
    "    !python scripts/run_calibration.py \\\n",
    "        --model meta-llama/Meta-Llama-3-8B-Instruct \\\n",
    "        --n-samples 512 \\\n",
    "        --output calibration_config.json\n",
    "    import json\n",
    "    cfg = json.load(open('calibration_config.json'))\n",
    "    print(f'\\n✓ Calibrated {len(cfg[\"layers\"])} layers')\n",
], "cell4"))

# ── CELL 5: Smoke Test ────────────────────────────────────────────────────
cells.append(code([
    "# CELL 5 — Smoke test (~5 min). Verifies ECC cache works end-to-end.\n",
    "import os; os.system('git -C /content/ecc-kv-cache pull origin main')\n",
    "!python scripts/smoke_test.py \\\n",
    "    --model meta-llama/Meta-Llama-3-8B-Instruct \\\n",
    "    --ctx-len 2000 \\\n",
    "    --n-tokens 50\n",
], "cell5"))

# ── CELL 6: VRAM Benchmark ────────────────────────────────────────────────
cells.append(code([
    "# CELL 6 — VRAM benchmark (~20 min)\n",
    "import os; os.system('git -C /content/ecc-kv-cache pull origin main')\n",
    "!python benchmarks/vram_reduction.py \\\n",
    "    --model meta-llama/Meta-Llama-3-8B-Instruct \\\n",
    "    --ctx-lengths 8000 32000 64000 128000 \\\n",
    "    --output results/vram_results.json\n",
], "cell6"))

# ── CELL 7: Throughput ────────────────────────────────────────────────────
cells.append(code([
    "# CELL 7 — Throughput benchmark (~15 min)\n",
    "import os; os.system('git -C /content/ecc-kv-cache pull origin main')\n",
    "!python benchmarks/throughput.py \\\n",
    "    --model meta-llama/Meta-Llama-3-8B-Instruct \\\n",
    "    --ctx-len 64000 \\\n",
    "    --n-tokens 200 \\\n",
    "    --output results/throughput.json\n",
], "cell7"))

# ── CELL 8: NIAH ──────────────────────────────────────────────────────────
cells.append(code([
    "# CELL 8 — NIAH benchmark (~4.5 hours, 100 trials)\n",
    "# Crash-safe: if interrupted, re-run this cell and it resumes automatically.\n",
    "import os; os.system('git -C /content/ecc-kv-cache pull origin main')\n",
    "!python benchmarks/niah_128k.py \\\n",
    "    --model meta-llama/Meta-Llama-3-8B-Instruct \\\n",
    "    --ctx-lengths 8000 32000 64000 128000 \\\n",
    "    --depths 0.0 0.25 0.5 0.75 1.0 \\\n",
    "    --trials 100 \\\n",
    "    --methods fp16 int4_bnb int4_ecc \\\n",
    "    --output results/niah_results.json\n",
], "cell8"))

# ── CELL 9: Generate Table ────────────────────────────────────────────────
cells.append(code([
    "# CELL 9 — Generate comparison table from results\n",
    "!python benchmarks/compare_baselines.py \\\n",
    "    --vram results/vram_results.json \\\n",
    "    --niah results/niah_results.json \\\n",
    "    --throughput results/throughput.json \\\n",
    "    --output results/comparison_table.md\n",
    "print(open('results/comparison_table.md').read())\n",
], "cell9"))

# ── CELL 10: Push Results ─────────────────────────────────────────────────
# Uses gh_token Colab secret for push auth.
# Add a secret named 'gh_token' with a GitHub PAT (repo scope) in Colab Secrets.
cells.append(code([
    "# CELL 10 — Push all results to GitHub\n",
    "# Requires Colab secret 'gh_token' = GitHub PAT with repo scope\n",
    "from google.colab import userdata\n",
    "import os, subprocess\n",
    "\n",
    "gh_token = userdata.get('gh_token')\n",
    "repo_url = f'https://{gh_token}@github.com/j-arndt/ecc-kv-cache.git'\n",
    "os.system('git config user.email \"j-arndt@users.noreply.github.com\"')\n",
    "os.system('git config user.name \"j-arndt\"')\n",
    "os.system(f'git remote set-url origin \"{repo_url}\"')\n",
    "\n",
    "# Force-add outputs even if gitignored\n",
    "os.system('git add -f calibration_config.json')\n",
    "os.system('git add -f results/')\n",
    "os.system('git commit -m \"results: A100 80GB benchmark — VRAM, NIAH 100-trial, throughput\"')\n",
    "os.system('git tag v0.1.0 --force')\n",
    "result = os.system('git push origin main --tags')\n",
    "if result == 0:\n",
    "    print('\\n✓ Done! Results at https://github.com/j-arndt/ecc-kv-cache')\n",
    "else:\n",
    "    print('\\n⚠ Push failed. Check that gh_token secret has repo scope.')\n",
], "cell10"))

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {
            "provenance": [],
            "machine_shape": "hm",
            "gpuType": "A100"
        },
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU"
    },
    "cells": cells
}

with open("notebooks/colab_a100.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Notebook written:", len(cells), "cells")
