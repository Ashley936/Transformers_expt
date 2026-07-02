import math
import torch
import torch.nn as nn
from torch import Tensor

# Creating text encodings
class TextEncoding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size, d_model) 
    
    def forward(self, x):
        return self.embedding(x)*math.sqrt(self.d_model) # (dim of x, d_model)

# Create positional encodings
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len:int, dropout:float):
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)
        # will be registering pe at the end as register buffer
        pe=torch.zeros(seq_len, d_model)
        pos=torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # [seq_len, 1] [[0], [1], [2]]
        ''' 
            we use torch.arange(0, d_model, 2) because in paper divide the positions into 2 sets
            0 and 1 will have i=0, 2 and 3 will have i=1, and so on
            also A^B = e^(ln(A)B)
            div_term shape : [d_model/2] ; ex: [1.0000, 0.9647, 0.9306, 0.8977, 0.8660] 
        '''
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Applying sin to even pos
        pe[:, 0::2] = torch.sin(pos*div_term) # [seq_len, d_mode/2]
        # Applying cos to odd pos
        pe[:, 1::2] = torch.cos(pos*div_term)

        # Add a batch dim (seq_len, d_model) => (batch_size, seq_len, d_model)
        pe=pe.unsqueeze(0)

        # tensor that is not a learning parameter but need to save when model is saved
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # x => (batch_size, seq_len, d_model)
        return self.dropout(x)


class LayerNormalisation(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.beta = nn.Parameter(torch.zeros(1)) # Added

    def forward(self, x: Tensor):
        mu = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha*(x-mu)/(std+self.eps) + self.beta

class FeedForwardBlock(nn.Module):
    '''
    Input tensor : (Batch Size, Sequence Length, d_model)
    Two linear transformations with a ReLU activation in between. (521 -> 2048 -> RELU -> 512)
    FFN(x) = max(0, W1.x + b1)W2 + b2
    This block is applied pointwise.
    It treats every single token in the sequence completely independently. 
    There is absolutely no communication between token at index 1 and token at index 2 inside this block.
    '''
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1=nn.Linear(d_model, d_ff)
        self.dropout=nn.Dropout(dropout)
        self.linear_2=nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: int):
        super().__init__()
        self.d_model=d_model
        self.h=h
        assert d_model%h==0, "d_model is not completely divisible by number of heads"
        self.d_k=d_model//h

        self.w_q=nn.Linear(d_model, d_model)
        self.w_k=nn.Linear(d_model, d_model)
        self.w_v=nn.Linear(d_model, d_model)
        self.w_o=nn.Linear(d_model, d_model)
        self.dropout=nn.Dropout(dropout)
    
    @staticmethod #That means you can call this method without instance of this class
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k=query.shape[-1]
        # (batch, h, seq_len, d_k)*(batch, h, d_k, seq_len) ----> (batch, h, seq_len, seq_len)
        attention_scores=query @ key.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask==0, -1e9)
        attention_scores=attention_scores.softmax(dim=-1) # (batch_size, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores=dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor):
        query=self.w_q(q)
        key=self.w_k(k)
        value=self.w_v(v)

        # (batch, seq_len, d_model) ----> (batch, seq_len, h, d_k) ---> (batch, h, seq_len, d_k) 
        '''
        doing (batch, seq_len, h, d_k) ---> (batch, h, seq_len, d_k)
        alows each head to take info from all the seq tokens
        {transpose(1, 2) applies transpose on the seq_len x h matrix}
        '''
        query=query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key=key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value=value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention for each head
        x, self.attention_scores=MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        # x : (batch, h, seq_len, d_k) -----> (batch, seq_len, h, d_k)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h*self.d_k)

        return self.w_o(x)

class ResidualConnection(nn.Module):
    '''Solves the The Vanishing Gradient Problem'''
    def __init__(self, d_model: int, dropout: float, norm_type: str):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.norm=LayerNormalisation(d_model)
        self.norm_type = norm_type
    
    def forward(self, x, sublayer):
        if self.norm_type == "pre": # Modern approach
            return x+self.dropout(sublayer(self.norm(x)))
        else: # From the paper
            return self.norm(x + self.dropout(sublayer(x)))


'''
This is internal Initialization approach: 
    1. Harder to swap attention
    2. Harder to test in isolation
class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout:float, h: int):
        super().__init__()
        self.mha=MultiHeadAttention(d_model, h, dropout)
        self.add_norm1=ResidualConnection(dropout)
        self.ff=FeedForwardBlock(d_model, d_ff, dropout)
        self.add_norm2=ResidualConnection(dropout)
    
    def forward(self, x: Tensor, src_mask: Tensor = None):
        x=self.add_norm1(x, lambda x: self.mha(x, x, x, src_mask))
        return self.add_norm2(x, self.ff)'''

'''This is dependency injection approach'''
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, ff_block: FeedForwardBlock, d_model: int, dropout: float, norm_type:str):
        super().__init__()
        self.self_attention_block=self_attention_block
        self.ff_block=ff_block
        self.residual_connections=nn.ModuleList([ResidualConnection(d_model, dropout, norm_type) for _ in range(2)])

    def forward(self, x: Tensor, src_mask: Tensor=None):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.ff_block)
        return x

'''
If we initialize layer inside encoder : Every layer shares the same hyperparameters and same class.
This dependency injection approach provides extreme flexibility
'''
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalisation()
    
    def forward(self, x: Tensor, mask: Tensor=None):
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attn_block: MultiHeadAttention, cross_attn_block: MultiHeadAttention, ff_block: FeedForwardBlock, d_model: int, dropout: float, norm_type:str):
        super().__init__()
        self.self_attn=self_attn_block
        self.cross_attn=cross_attn_block
        self.ff_block=ff_block

        self.residual_connections=nn.ModuleList([ResidualConnection(d_model, dropout, norm_type) for _ in range(3)])

    def forward(self, x: Tensor, enc_out: Tensor, src_mask: Tensor=None, tgt_mask: Tensor=None):
        x=self.residual_connections[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x=self.residual_connections[1](x, lambda x: self.cross_attn(x, enc_out, enc_out, src_mask))
        x=self.residual_connections[2](x, self.ff_block)

        return x
    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalisation()
    
    def forward(self, x: Tensor, enc_out: Tensor, src_mask: Tensor, tgt_mask: Tensor):
        for layer in self.layers:
            x=layer(x, enc_out, src_mask, tgt_mask)
        return self.norm(x)



class ProjectionLayer(nn.Module):
    '''Project decoder output to vocab size space
        (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
    '''
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj=nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        return self.proj(x)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_emb: TextEncoding, tgt_emb: TextEncoding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection: ProjectionLayer):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_emb=src_emb
        self.tgt_emb=tgt_emb
        self.src_pos=src_pos
        self.tgt_pos=tgt_pos
        self.projection_layer=projection
    
    def encode(self, x, src_mask):
        x=self.src_emb(x)
        x=self.src_pos(x)
        return self.encoder(x, src_mask)
    
    def decode(self, x, enc_out, src_mask, tgt_mask):
        x=self.tgt_emb(x)
        x=self.tgt_pos(x)
        return self.decoder(x, enc_out, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)



def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, norm_type: str = "pre", N: int=6, h: int=8, d_ff: int=2048, dropout: float=0.1):
    # Initialize text embeddings
    src_emb=TextEncoding(d_model, src_vocab_size)
    tgt_emb=TextEncoding(d_model, tgt_vocab_size)

    # Initialize pos embeddings
    src_pos_emb=PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos_emb=PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Initialize encoder layers
    encoder_layers=[]
    for _ in range(N):
        self_attention_block=MultiHeadAttention(d_model, h, dropout)
        ff_block=FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block=EncoderBlock(self_attention_block, ff_block, d_model, dropout, norm_type)
        encoder_layers.append(encoder_block)
    
    # Initialize decoder layers
    decoder_layers=[]
    for _ in range(N):
        self_attention_block=MultiHeadAttention(d_model, h, dropout)
        cross_attention_block=MultiHeadAttention(d_model, h, dropout)
        ff_block=FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block=DecoderBlock(self_attention_block, cross_attention_block, ff_block, d_model, dropout, norm_type)
        decoder_layers.append(decoder_block)
    
    # Initialize the projection layer
    projection_layer=ProjectionLayer(d_model, tgt_vocab_size)

    # Initialize encoder and decoder
    encoder=Encoder(nn.ModuleList(encoder_layers))
    decoder=Decoder(nn.ModuleList(decoder_layers))

    # Initialize transformer
    transformer=Transformer(encoder, decoder, src_emb, tgt_emb, src_pos_emb, tgt_pos_emb, projection_layer)

    # Initialize weights in transformer
    for w in transformer.parameters():
        if w.dim() > 1:
            nn.init.xavier_uniform_(w)
    
    return transformer
        