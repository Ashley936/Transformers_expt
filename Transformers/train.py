import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from tqdm import tqdm
from torchmetrics.text import BLEUScore
from torch.optim.lr_scheduler import LambdaLR

from dataset import BillingualDataset, causal_mask
from model import build_transformer
from config import get_weights_file_path, get_config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def greedy_decode(model, enc_input, enc_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx=tokenizer_src.token_to_id("[sos]")
    eos_idx=tokenizer_src.token_to_id("[eos]")

    # Precompute the enc output and use it to find each decoder token
    enc_output=model.encode(enc_input, enc_mask) # (batch=1, src_seq_len, d_model)

    decoder_input=torch.empty(1, 1).fill_(sos_idx).type_as(enc_input).to(device) # [[sos_id]] (1, 1)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build causal mask (hide i+1th to nth token for each ith token from 1 to n where n is the last added token)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(enc_mask).to(device)

        out = model.decode(decoder_input, enc_output, enc_mask, decoder_mask) # (batch, n+1, d_model) (1, 1, 512) first time

        # Get the last token
        prob = model.project(out[:, -1]) # (takes the last token (1, 512) ----> (1, vocab_size))

        _, next_word = torch.max(prob, dim=1) # next_word shape (1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(enc_input).fill_(next_word.item()).to(device)], dim=1)
        # [[sos_id, token_1, token_2, .... token_n+1]]
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0) # strips away the dummy batch dimension --> (num_tokens_gen)



def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, validation_batch_size=2):
    # change model to eval mode
    model.eval()
    count = 0

    # Lists to store text for BLEU calculation
    expected = []
    predicted = []

    # Size of control window
    console_width = 80 

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['enc_input'].to(device) # (B, seq_len)
            encoder_mask = batch['enc_mask'].to(device) # (B, 1, 1, seq_len)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len=60, device=device)
            
            src_text = batch['src_text'][0]
            tgt_text = batch['tgt_text'][0]
            output_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # Accumulate texts for BLEU (BLEUScore expects targets as a list of lists)
            expected.append([tgt_text])
            predicted.append(output_text)
            if count < 2:
                print_msg('-'*console_width)
                print_msg(f'SOURCE TEXT: {src_text}')
                print_msg(f'EXPECTED TEXT: {tgt_text}')
                print_msg(f'MODEL OUTPUT TEXT: {output_text}')

            if count == validation_batch_size:
                break
    
    # Calculate and log the BLEU Score
    metric = BLEUScore()
    bleu_score = metric(predicted, expected)
    
    writer.add_scalar('validation BLEU', bleu_score.item(), global_step)

    return bleu_score.item()
    
# This is a generator (the func does not completes on the first func call)
def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path=Path(config['tokenizer_path'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer=Tokenizer(WordLevel(unk_token="[unk]"))
        tokenizer.pre_tokenizer=Whitespace()
        trainer=WordLevelTrainer(special_tokens=["[unk]", "[pad]", "[sos]", "[eos]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer=Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # Get only the train dataset and then create test, train, valid set
    ds_raw=load_dataset('Helsinki-NLP/opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')
    set_seed(config.get('seed', 42))
    # Building tokenizers
    tokenizer_src=get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt=get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    # Train and test split 9:1
    train_ds_size = int(0.9*len(ds_raw))
    val_ds_size = len(ds_raw)-train_ds_size
    '''You can initialize a custom generator 
        generator1 = torch.Generator().manual_seed(42)'''
    seed=config['seed']
    train_ds_raw, val_ds_raw=random_split(ds_raw, [train_ds_size, val_ds_size], generator=torch.Generator().manual_seed(seed))

    train_ds=BillingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    val_ds=BillingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    
    # Now we check the max length of src or tgt lang in dataset to find the optimal seq len

    max_src_len=0
    max_tgt_len=0

    for item in ds_raw:
        src_ids=tokenizer_src.encode(item["translation"][config["lang_src"]])
        tgt_ids=tokenizer_tgt.encode(item["translation"][config["lang_tgt"]])
        max_src_len=max(max_src_len, len(src_ids))
        max_tgt_len=max(max_tgt_len, len(tgt_ids))
    
    print(f"Max length of source sequence: {max_src_len}")
    print(f"Max length of target sequence: {max_tgt_len}")

    # Create Dataloaders from dataset
    train_dataloader=DataLoader(train_ds, config["batch_size"], shuffle=True)
    val_dataloader=DataLoader(val_ds, batch_size=1, shuffle=True) # process each sentence 1 by 1

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, src_vocab_size, tgt_vocab_size):
    model=build_transformer(src_vocab_size, tgt_vocab_size, config['seq_len'], config['seq_len'], config['d_model'], config['norm_type'])
    return model

def rate(step, warmup):
    if step == 0:
        step = 1
    # Peaks at 1.0 when step == warmup, matching flat LR at that point
    return min(step ** -0.5, step * warmup ** -1.5) / (warmup ** -0.5)


def train(config, train_dataloader=None, val_dataloader=None, tokenizer_src=None, tokenizer_tgt=None):
    # define device
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)

    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    if train_dataloader is None or val_dataloader is None or tokenizer_src is None or tokenizer_tgt is None:
        train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt=get_ds(config)

    set_seed(config.get('seed', 42))
    model=get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Create Tensorboard
    writer=SummaryWriter(config['experiment_name'])

    optimizer=torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1e-9)
    if config['lr_schedule'] == 'warmup':
        lr_scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: rate(step, warmup=config["warmup_steps"]),
        )
        print(f"[warmup] enabled, warmup_steps={config['warmup_steps']}")
    else:
        lr_scheduler = None
        print("[warmup] disabled — flat LR")

    
    initial_epoch=0
    global_step=0
    best_bleu=-1
    best_checkpoint_path=None
    previous_best_filename=None

    # Metrics tracked for ablation analysis (loss curve + gradient norms + val BLEU).
    # Dumped to disk periodically so an overnight Colab disconnect doesn't lose progress.
    history = {"train": [], "val": []}
    history_path = Path(f"{config['datasource']}_{config['model_folder']}") / "history.json"

    # Load a crashed training from latest .pt checkpoint
    if config['preload']:
        model_path=get_weights_file_path(config, config['preload']) 
        print(f'Loading pretrained model from path {model_path}')
        state=torch.load(model_path, map_location=device) # BY DEFAULT LOADS MODEL ON GPU

        # load model weights 
        model.load_state_dict(state['model_state_dict'])
        initial_epoch=state['epoch']+1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step=state['global_step']
    

    ''' Label smoothing : We take some part of most probable token and distribute it to other tokens
    This makes the model less over-confident. '''
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[pad]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        batch_i = 0
        for batch in batch_iterator:
            batch_i += 1
            '''batch -> enc_input, dec_input, label, enc_mask, dec_mask, src_txt, tgt_txt'''

            encoder_input = batch['enc_input'].to(device) # (B, seq_len)
            decoder_input = batch['dec_input'].to(device) # (B, seq_len)
            encoder_mask = batch['enc_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['dec_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the transformer
            encoder_output=model.encode(encoder_input, encoder_mask) #(B, seq_len, d_model)
            decoder_output=model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask) #(B, seq_len, d_model)
            proj_out = model.project(decoder_output) # (B, seq_len, tgt_vocab_size)

            label = batch['label'].to(device) # (B, seq_len)

            # Calculate the loss 
            ''' .view(-1) ->squash all dim 
                (B, seq_len) => (B*seq_len)
                (B, seq_len, vocab_size) => (B*seq_len, vocab_size)'''
            loss = loss_fn(proj_out.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

            if config['lr_schedule'] == 'warmup':
                batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}", "lr": f"{lr_scheduler.get_last_lr()[0]:6.1e}", "batch step": f"{batch_i:03d}"})
            else:
                batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # log the loss in tensorboard
            writer.add_scalar('train loss', loss.item(), global_step)
            

            # Backpropagate the loss
            loss.backward()
            # increased so that no clipping happens
            unclipped_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)

            writer.add_scalar('Gradient Norm (Unclipped)', unclipped_grad_norm.item(), global_step)

            #update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)  # prevents the system from doing unnecessary memory allocations for zeros, marginally speeding up training loop
            if config['lr_schedule'] == 'warmup':
                lr_scheduler.step()

            # print lr every 10 step
            if global_step % 10 == 0:
                current_lr = lr_scheduler.get_last_lr()[0] if config['lr_schedule'] == 'warmup' else config['lr']
                writer.add_scalar('learning rate', current_lr, global_step)
                writer.flush() # write to disk
            
            history["train"].append({
                "step": global_step,
                "epoch": epoch,
                "loss": loss.item(),
                "grad_norm": unclipped_grad_norm.item(),
                "lr": lr_scheduler.get_last_lr()[0] if config['lr_schedule'] == 'warmup' else config['lr'],
            })

            # Run validation
            if global_step != 0 and global_step % config['val_interval'] == 0:
                current_bleu = run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer, config["val_batch_size"])
                history["val"].append({"step": global_step, "epoch": epoch, "bleu": current_bleu})

                # Persist history to disk after every validation pass (cheap, and this is
                # the checkpoint an overnight run can least afford to lose).
                with open(history_path, "w") as f:
                    json.dump(history, f, indent=2)

                # Check if this is the best model so far
                if current_bleu > best_bleu:
                    best_bleu = current_bleu
                    batch_iterator.write(f"New best BLEU score: {best_bleu:6.3f}")

                    if config.get('save_weights', True):
                        best_model_filename = get_weights_file_path(config, f'{epoch}-{batch_i}')
                        torch.save({
                            'epoch': epoch, 
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'global_step': global_step,
                            'best_bleu': best_bleu # Save the score so you know how good it is later
                        }, best_model_filename)
                        if previous_best_filename is not None and Path(previous_best_filename).exists():
                            Path(previous_best_filename).unlink(missing_ok=True)
                        previous_best_filename = best_model_filename
                        best_checkpoint_path = best_model_filename
            global_step+=1
        

        ''' 
        # Save model at each epoch
        model_filename = get_weights_file_path(config, epoch)
        torch.save({
            'epoch': epoch, 
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
        '''

    # Final history flush + summary returned to caller (e.g. the ablation runner),
    # so it doesn't have to re-parse tensorboard logs to compare runs.
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    writer.close()

    return {
        "history": history,
        "best_bleu": best_bleu,
        "best_checkpoint_path": best_checkpoint_path,
        "history_path": str(history_path),
        "config": config,
    }


if __name__ == '__main__':
    # To remove warnings : warnings.filterwarnings('ignore')

    config = get_config()
    train(config)