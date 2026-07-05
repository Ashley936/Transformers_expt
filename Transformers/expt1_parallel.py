"""
Ablation Study Runner (2x T4 parallel version)
================================================
Pre-LN vs Post-LN  x  Flat LR vs Warmup+Decay  (2x2 grid, 4 runs)

Now split 2 runs per GPU instead of 4 runs sequential on 1 GPU, roughly
halving total wall-clock time for the sweep.

New for the parallel version:

- The grid is split into two halves, one per GPU. Each half runs in its own
  process, pinned to one GPU via CUDA_VISIBLE_DEVICES.

- Each process builds its OWN dataloaders/tokenizers. Sharing dataloader
  objects across process boundaries isn't safe, so the "build once, reuse
  across all runs" optimization from the sequential version is dropped -
  a small one-time cost per process, negligible against total sweep time.

- Each process writes its own results file (results_gpu0.json /
  results_gpu1.json) DURING the sweep, to avoid two processes writing the
  same shared JSON with no lock and corrupting it. The two files are merged
  into ablation_results.json only after BOTH processes finish.

- CUDA_VISIBLE_DEVICES must be set, and torch must be imported, INSIDE each
  worker process - not at module level - or the pinning won't take effect.

- multiprocessing must use 'spawn', not the default 'fork', for CUDA to work
  correctly across processes.
"""
import copy
import gc
import json
import os
import time
import traceback
from pathlib import Path

import torch.multiprocessing as mp

DRIVE_ROOT = Path("/kaggle/working/transformer_ablation")
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Ablation-specific overrides
# ---------------------------------------------------------------------------
ABLATION_EPOCHS = 10
ABLATION_WARMUP_STEPS = 500
ABLATION_VAL_INTERVAL = 600     # less validation overhead
ABLATION_VAL_BATCH_SIZE = 10    # less validation overhead

GRID = [
    {"norm_type": "pre",  "lr_schedule": "flat"},
    {"norm_type": "pre",  "lr_schedule": "warmup"},
    {"norm_type": "post", "lr_schedule": "flat"},
    {"norm_type": "post", "lr_schedule": "warmup"},
]

# Split 2+2, keeping one Pre-LN and one Post-LN cell per GPU rather than
# splitting by norm_type or schedule - so a GPU-specific quirk (thermal
# throttling, driver issue) doesn't confound one whole axis of the ablation.
GRID_HALVES = [GRID[0::2], GRID[1::2]]  # [[pre/flat, post/flat], [pre/warmup, post/warmup]]


# ---------------------------------------------------------------------------
# 2. Worker: runs its assigned cells sequentially on ONE pinned GPU
# ---------------------------------------------------------------------------
def run_worker(gpu_id, cells, base_config_dict):
    # MUST be set before torch is imported in this process, or the pinning
    # silently does nothing and both processes fight over both GPUs.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    from config import get_config  # noqa: F401 (kept for parity with base_config_dict shape)
    from train import train, get_ds

    print(f"[GPU {gpu_id}] Building dataloaders/tokenizers for this worker's runs...")
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(base_config_dict)

    worker_results_path = DRIVE_ROOT / f"results_gpu{gpu_id}.json"
    worker_results = {}
    if worker_results_path.exists():
        print(f"[GPU {gpu_id}] Found existing partial results, loading state...")
        with open(worker_results_path, "r") as f:
            worker_results = json.load(f)

    for cell in cells:
        run_name = f"{cell['norm_type']}LN_{cell['lr_schedule']}"

        if worker_results.get(run_name, {}).get("status") == "completed":
            print(f"[GPU {gpu_id}] SKIPPING RUN: {run_name} (Already Completed)")
            continue

        print(f"\n{'=' * 70}\n[GPU {gpu_id}] STARTING RUN: {run_name}\n{'=' * 70}")

        config = copy.deepcopy(base_config_dict)
        config.update(cell)
        config["num_epochs"] = ABLATION_EPOCHS
        config["warmup_steps"] = ABLATION_WARMUP_STEPS
        config["val_interval"] = ABLATION_VAL_INTERVAL
        config["val_batch_size"] = ABLATION_VAL_BATCH_SIZE
        config["preload"] = ""

        config["experiment_name"] = str(DRIVE_ROOT / "runs" / run_name)
        config["model_folder"] = f"weights_{run_name}"
        config["datasource"] = str(DRIVE_ROOT / "checkpoints")

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
            print(f"[GPU {gpu_id}] RUN {run_name} completed. Best BLEU: {result['best_bleu']:.3f}, "
                  f"checkpoint: {result['best_checkpoint_path']}")
        except Exception as e:
            print(f"[GPU {gpu_id}] RUN {run_name} FAILED: {e}")
            traceback.print_exc()
            result = {"status": "failed", "error": str(e), "config": config}

        result["duration_sec"] = time.time() - start
        worker_results[run_name] = result

        # Persist after every run - protects this worker's progress from a
        # mid-sequence disconnect independently of the other worker.
        with open(worker_results_path, "w") as f:
            json.dump(worker_results, f, indent=2)

        del result
        gc.collect()
        torch.cuda.empty_cache()

    print(f"[GPU {gpu_id}] Worker finished all assigned runs.")


# ---------------------------------------------------------------------------
# 3. Launch both workers in parallel, then merge results
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from config import get_config

    base_config = get_config()
    base_config["batch_size"] = 24
    base_config["val_interval"] = ABLATION_VAL_INTERVAL

    mp.set_start_method("spawn", force=True)

    processes = []
    for gpu_id, cells in enumerate(GRID_HALVES):
        p = mp.Process(target=run_worker, args=(gpu_id, cells, base_config))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\nBoth GPU workers finished. Merging results...")

    merged_results = {}
    for gpu_id in range(len(GRID_HALVES)):
        worker_results_path = DRIVE_ROOT / f"results_gpu{gpu_id}.json"
        if worker_results_path.exists():
            with open(worker_results_path) as f:
                merged_results.update(json.load(f))

    results_path = DRIVE_ROOT / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(merged_results, f, indent=2)

    print(f"All runs finished. Merged results saved to {results_path}")

    # -----------------------------------------------------------------------
    # 4. Comparison plots (identical to the sequential version - reads from
    # the merged ablation_results.json, doesn't care how the runs were produced)
    # -----------------------------------------------------------------------
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
        axes[0, 1].set_yscale("log")

        axes[1, 0].set_title("Validation BLEU")
        axes[1, 0].set_xlabel("step"); axes[1, 0].set_ylabel("BLEU"); axes[1, 0].legend()

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