from pathlib import Path

def get_config():
    return {
        "batch_size": 8, # 24 for google collab  steps -> 1200
        "num_epochs": 20,
        "lr": 10**-4, # usually we take large lr and the reduce it overtime
        "warmup_steps": 1000, # 300 for collab
        "save_weights": True,
        "seq_len": 350,
        "d_model": 512,
        "datasource": 'opus_books',
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "",
        "tokenizer_path": "dataset/tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "val_interval": 900, # 300 for collab
        "val_batch_size": 50
    }
    # in the expt1.py -> "norm_type": "pre", # 'pre' or 'post' FOR EXPT
    #                    "lr_schedule": "flat", # 'flat' or 'warmup' FOR EXPT

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])