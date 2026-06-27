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

from dataset import BillingualDataset, causal_mask
from model import build_transformer
from config import get_weights_file_path, get_config

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
    ds_raw=load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    # Building tokenizers
    tokenizer_src=get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt=get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    # Train and test split 9:1
    train_ds_size = int(0.9*len(ds_raw))
    val_ds_size = len(ds_raw)-train_ds_size
    '''You can initialize a custom generator 
        generator1 = torch.Generator().manual_seed(42)'''
    train_ds_raw, val_ds_raw=random_split(ds_raw, [train_ds_size, val_ds_size])

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
    model=build_transformer(src_vocab_size, tgt_vocab_size, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train(config):
    # define device
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt=get_ds(config)
    model=get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())

    # Create Tensorboard
    writer=SummaryWriter(config['experiment_name'])

    optimizer=torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    
    initial_epoch=0
    global_step=0

    # Load a crashed training from latest .pt checkpoint
    if config['preload']:
        model_path=get_weights_file_path(config, config['preload']) 
        print(f'Loading pretrained model from path {model_path}')
        state=torch.load(model_path) # BY DEFAULT LOADS MODEL ON GPU
        initial_epoch=state['epoch']+1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step=state['global_step']
    

    ''' Label smoothing : We take some part of most probable token and distribute it to other tokens
    This makes the model less over-confident. '''
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[pad]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')

        for batch in batch_iterator:
            '''batch -> enc_input, dec_input, label, enc_mask, dec_mask, src_txt, tgt_txt'''
            encoder_input = batch['enc_input'].to(device) # (B, seq_len)
            decoder_input = batch['dec_input'].to(device) # (B, seq_len)
            encoder_mask = batch['enc_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['dec_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the transformer
            ecoder_output=model.encode(encoder_input, encoder_mask) #(B, seq_len, d_model)
            decoder_output=model.decode(decoder_input, ecoder_output, encoder_mask, decoder_mask) #(B, seq_len, d_model)
            proj_out = model.project(decoder_output) # (B, seq_len, tgt_vocab_size)

            label = batch['label'].to(device) # (B, seq_len)

            # Calculate the loss 
            ''' .view(-1) ->squash all dim 
                (B, seq_len) => (B*seq_len)
                (B, seq_len, vocab_size) => (B*seq_len, vocab_size)'''
            loss = loss_fn(proj_out.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # log the loss in tensorboard
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush() # write to disk

            # Backpropagate the loss
            loss.backward()

            #update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step+=1
        
        # Save model at each epoch
        model_filename = get_weights_file_path(config, epoch)
        torch.save({
            'epoch': epoch, 
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    # To remove warnings : warnings.filterwarnings('ignore')

    config = get_config()
    train(config)





