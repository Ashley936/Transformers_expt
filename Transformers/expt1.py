"""
Ablation Study Runner
======================
Pre-LN vs Post-LN  x  Flat LR vs Warmup+Decay  (2x2 grid, 4 runs, sequential)
"""
import copy
import gc
import json
import time
import traceback
from pathlib import Path

import torch

from config import get_config
from train import train, get_ds

DRIVE_ROOT = Path("/kaggle/working/transformer_ablation")
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Ablation-specific overrides
# ---------------------------------------------------------------------------

ABLATION_EPOCHS = 5
ABLATION_WARMUP_STEPS = 300
ABLATION_VAL_INTERVAL = 600     # less validation overhead
ABLATION_VAL_BATCH_SIZE = 10    # less validation overhead

GRID = [
    {"norm_type": "pre",  "lr_schedule": "flat"},
    {"norm_type": "pre",  "lr_schedule": "warmup"},
    {"norm_type": "post", "lr_schedule": "flat"},
    {"norm_type": "post", "lr_schedule": "warmup"},
]

# ---------------------------------------------------------------------------
# 2. Build the dataset / tokenizers ONCE, shared across all 4 runs
# ---------------------------------------------------------------------------
base_config = get_config()
base_config["batch_size"] = 24
base_config["val_interval"] = ABLATION_VAL_INTERVAL
print("Building shared dataloaders/tokenizers (reused by all 4 runs)...")
train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(base_config)

# ---------------------------------------------------------------------------
# 3. Run the grid sequentially
# ---------------------------------------------------------------------------
all_results = {}
results_path = DRIVE_ROOT / "ablation_results.json"

if results_path.exists():
    print(f"Found existing results at {results_path}. Loading state...")
    with open(results_path, "r") as f:
        all_results = json.load(f)

for cell in GRID:
    run_name = f"{cell['norm_type']}LN_{cell['lr_schedule']}"

    if all_results.get(run_name, {}).get("status") == "completed":
        print(f"\n{'=' * 70}\nSKIPPING RUN: {run_name} (Already Completed)\n{'=' * 70}")
        continue

    print(f"\n{'=' * 70}\nSTARTING RUN: {run_name}\n{'=' * 70}")

    config = copy.deepcopy(base_config)
    config.update(cell)
    config["num_epochs"] = ABLATION_EPOCHS
    config["val_interval"] = ABLATION_VAL_INTERVAL
    config["preload"] = ""  # each ablation cell trains from scratch

    # Route logs/checkpoints straight to drive/kaggle (separate tensorboard dir + separate weights dir).
    config["experiment_name"] = str(DRIVE_ROOT / "runs" / run_name)
    config["model_folder"] = f"weights_{run_name}"
    config["datasource"] = str(DRIVE_ROOT / "checkpoints")

    # FOR Kaggle
    config["warmup_steps"] = ABLATION_WARMUP_STEPS
    config["val_batch_size"] = ABLATION_VAL_BATCH_SIZE
    config["val_interval"] = ABLATION_VAL_INTERVAL
    
    
    start = time.time()
    try:
        result = train(
            config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            tokenizer_src=tokenizer_src,
            tokenizer_tgt=tokenizer_tgt,
        )
        result["status"] = "completed"
        print(f"RUN {run_name} completed. Best BLEU: {result['best_bleu']:.3f}, "
              f"checkpoint: {result['best_checkpoint_path']}")
    except Exception as e:
        print(f"RUN {run_name} FAILED: {e}")
        traceback.print_exc()
        result = {"status": "failed", "error": str(e), "config": config}

    result["duration_sec"] = time.time() - start
    all_results[run_name] = result

    # Persist after EVERY run, not just at the end
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Free GPU memory before the next run
    del result
    gc.collect()
    torch.cuda.empty_cache()

print(f"\nAll runs finished. Results saved to {results_path}")

# ---------------------------------------------------------------------------
# 4. Comparison plots (loss curves, grad norms, per-cell + overlay)
# ---------------------------------------------------------------------------

try:
    import matplotlib.pyplot as plt

    with open(results_path) as f:
        all_results = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Ablation: Pre-LN vs Post-LN x Flat vs Warmup+Decay")

    for run_name, result in all_results.items():
        if result.get("status") != "completed":
            continue
        train_hist = result["history"]["train"]
        steps = [h["step"] for h in train_hist]
        losses = [h["loss"] for h in train_hist]
        grad_norms = [h["grad_norm"] for h in train_hist]
        val_hist = result["history"]["val"]
        val_steps = [h["step"] for h in val_hist]
        val_bleu = [h["bleu"] for h in val_hist]

        axes[0, 0].plot(steps, losses, label=run_name, alpha=0.8)
        axes[0, 1].plot(steps, grad_norms, label=run_name, alpha=0.8)
        axes[1, 0].plot(val_steps, val_bleu, label=run_name, marker="o", alpha=0.8)

    axes[0, 0].set_title("Train loss")
    axes[0, 0].set_xlabel("step"); axes[0, 0].set_ylabel("loss"); axes[0, 0].legend()

    axes[0, 1].set_title("Gradient norm (L2, unclipped)")
    axes[0, 1].set_xlabel("step"); axes[0, 1].set_ylabel("grad norm"); axes[0, 1].legend()
    axes[0, 1].set_yscale("log")  # Post-LN instability tends to show up as spikes/blowups

    axes[1, 0].set_title("Validation BLEU")
    axes[1, 0].set_xlabel("step"); axes[1, 0].set_ylabel("BLEU"); axes[1, 0].legend()

    # Simple stability summary: std/mean of grad norm over the back half of
    # training per run (a crude but useful "did this config stay stable?" number).
    axes[1, 1].axis("off")
    summary_lines = ["Stability summary (grad-norm std/mean, 2nd half of training):"]
    for run_name, result in all_results.items():
        if result.get("status") != "completed":
            summary_lines.append(f"  {run_name}: FAILED - {result.get('error')}")
            continue
        gn = [h["grad_norm"] for h in result["history"]["train"]]
        half = gn[len(gn) // 2:]
        if half:
            mean_gn = sum(half) / len(half)
            std_gn = (sum((x - mean_gn) ** 2 for x in half) / len(half)) ** 0.5
            cv = std_gn / mean_gn if mean_gn else float("nan")
            summary_lines.append(f"  {run_name}: mean={mean_gn:.2f}  cv={cv:.3f}  best_bleu={result['best_bleu']:.3f}")
    axes[1, 1].text(0, 1, "\n".join(summary_lines), va="top", family="monospace", fontsize=9)

    plt.tight_layout()
    plot_path = DRIVE_ROOT / "ablation_comparison.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Comparison plot saved to {plot_path}")
    plt.show()

except ImportError:
    print("matplotlib not available - skipping plots. Results are still in ablation_results.json.")