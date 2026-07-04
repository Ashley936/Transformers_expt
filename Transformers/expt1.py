"""
Ablation Study Runner
======================
Pre-LN vs Post-LN  x  Flat LR vs Warmup+Decay  (2x2 grid, 4 runs, sequential)

Design notes (why this file looks the way it does):

- model.py / config.py are untouched. train.py only got the minimum needed to
  support this: gradient-norm tracking, a returned metrics dict, best-only
  checkpointing, and the ability to accept pre-built dataloaders. Everything
  ablation-specific (the grid, run naming, Drive paths, plotting, crash
  isolation) lives HERE so your base transformer code stays reusable for
  future projects (KV-cache, RoPE, etc.) without ablation cruft in it.

- Tokenizers/dataloaders are built ONCE and reused across all 4 runs, since
  datasource/langs/seq_len/batch_size are identical in every cell of the
  grid. Rebuilding them 4x would waste a big chunk of your free-tier GPU time
  on pure tokenization/dataset-scanning.

- Every run is wrapped in try/except. If one config OOMs or crashes at 3am,
  the other 3 still run - you don't want one bad cell to kill an unattended
  overnight sweep.

- Results are flushed to Google Drive after EVERY run (not just at the end),
  because Colab free-tier runtimes can be reclaimed/disconnected without
  warning and local (non-Drive) storage is wiped when that happens.

- Only the single best checkpoint per run is kept (train.py handles the
  delete-old-then-save-new logic). You said you'll push to Drive yourself
  after each run - but since local storage disappears on disconnect, this
  script writes checkpoints straight to Drive to begin with, so there's
  nothing to lose even if you don't get to it in time.
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

# ---------------------------------------------------------------------------
# 0. Colab / Drive setup
# ---------------------------------------------------------------------------
# Uncomment when running in Colab. Mounting Drive FIRST is what makes this
# whole "survive a disconnect" strategy work - do not skip it.
#
# from google.colab import drive
# drive.mount('/content/drive')

DRIVE_ROOT = Path("/kaggle/working/transformer_ablation")
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Ablation-specific overrides
# ---------------------------------------------------------------------------
# Free-tier T4 sessions are capped (~12h, often much less with idle timeouts),
# and this study cares about TRAINING DYNAMICS (loss curves, grad norms,
# convergence stability), not final translation quality. 20 epochs x 4 runs
# on the full opus_books en-it set will not finish overnight. Cut epochs down
# and validate more frequently so each run still produces a readable curve.
ABLATION_EPOCHS = 5              # <- raise/lower based on how much GPU time you actually get
ABLATION_VAL_INTERVAL = 300      # more frequent than the default 900, since runs are shorter

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
# For google collab
base_config["batch_size"] = 24
base_config["val_interval"] = 300
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

    # Route logs/checkpoints straight to Drive, and keep the 4 runs from
    # colliding with each other (separate tensorboard dir + separate weights dir).
    config["experiment_name"] = str(DRIVE_ROOT / "runs" / run_name)
    config["model_folder"] = f"weights_{run_name}"
    config["datasource"] = str(DRIVE_ROOT / "checkpoints")

    # FOR Kaggle
    config["warmup_steps"] = 300
    config["val_batch_size"] = 10
    config["val_interval"] = 600
    
    
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

    # Persist after EVERY run, not just at the end - this is the line that
    # protects the whole overnight sweep from a mid-sequence disconnect.
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Free GPU memory before the next run - 4 transformers back-to-back on a
    # free-tier T4 will OOM if you don't explicitly release the previous one.
    del result
    gc.collect()
    torch.cuda.empty_cache()

print(f"\nAll runs finished. Results saved to {results_path}")

# ---------------------------------------------------------------------------
# 4. Comparison plots (loss curves, grad norms, per-cell + overlay)
# ---------------------------------------------------------------------------
# Safe to re-run this section on its own later (e.g. next morning) by loading
# ablation_results.json instead of re-running the sweep - it doesn't depend
# on anything still being in memory.
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