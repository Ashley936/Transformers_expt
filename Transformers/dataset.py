import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BillingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        self.ds=ds
        self.seq_len=seq_len
        self.tokenizer_src=tokenizer_src
        self.tokenizer_tgt=tokenizer_tgt
        self.src_lang=src_lang
        self.tgt_lang=tgt_lang
        #convert start of sentence to a token id
        self.sos_token=torch.tensor([tokenizer_src.token_to_id("[sos]")], dtype=torch.int64)
        self.eos_token=torch.tensor([tokenizer_src.token_to_id("[eos]")], dtype=torch.int64)
        self.pad_token=torch.tensor([tokenizer_src.token_to_id("[pad]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        target_pair=self.ds[index]
        src_txt=target_pair['translation'][self.src_lang]
        tgt_txt=target_pair['translation'][self.tgt_lang]

        enc_input_token=self.tokenizer_src.encode(src_txt).ids
        dec_input_token=self.tokenizer_tgt.encode(tgt_txt).ids

        enc_pad_len=self.seq_len - len(enc_input_token) - 2 # subtract 2 for eos and sos token
        dec_pad_len=self.seq_len - len(dec_input_token) - 1 # only 1 of eos or sos token is present (label-> eos, dec input -> sos)

        if(enc_pad_len<0 or dec_pad_len<0):
            raise ValueError("Sentence length is longer that max seq len")
        
        enc_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_token, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token]*enc_pad_len, dtype=torch.int64)
        ], dim=0)

        dec_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_token, dtype=torch.int64),
            torch.tensor([self.pad_token]*dec_pad_len, dtype=torch.int64)
        ], dim=0)

        label = torch.cat([
            torch.tensor(dec_input_token, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token]*dec_pad_len, dtype=torch.int64),
            
        ], dim=0)

        # double check for seq len
        # print(f'Size of : Encoder input tensor {enc_input.size(0)} and decoder input tensor {dec_input.size(0)}')
        assert enc_input.size(0) == self.seq_len
        assert dec_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "enc_input": enc_input, # (seq_len)
            "dec_input": dec_input, # (seq_len)
            "label": label, # (seq_len)
            "enc_mask": (enc_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len) and pad token is masked
            "dec_mask": (dec_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(dec_input.size(0)), # output (1, seq_len, seq_len)
            "src_text": src_txt,
            "tgt_text": tgt_txt
        }
    
def causal_mask(size):
    '''size: sequence length'''
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask==0
    
    