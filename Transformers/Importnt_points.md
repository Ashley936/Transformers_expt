# Important questions I encountered while executing the paper

---

## Questions

- [Q1: How does Pre-LN and Post-LN in the FFN layer effect Gradient Flow and Learning Rate Sensitivity?](#q1-how-does-pre-ln-and-post-ln-in-the-ffn-layer-effect-gradient-flow-and-learning-Rate-Sensitivity)
- [Q2: Where is dropout applied? And how does modern approach change to applying dropout in FFN block?](#q2-where-is-dropout-applied-and-how-does-modern-approach-change-to-applying-dropout-in-ffn-block)
- [Q3: Should EncoderBlock accept attention/FF layers as constructor arguments, or initialize them internally?](#q3-should-encoderblock-accept-attentionff-layers-as-constructor-arguments-or-initialize-them-internally)
- [Q4: What is the difference between masking in cross-attention and self-attention?](#q4-what-is-the-difference-between-masking-in-cross-attention-and-self-attention)

---

## Knowledge points

1. Chosing batch size and stride during batch creation:

```
    Increasing bacth_size -> model update after more data -> less computing more memory (may lead to diminishing returns)
    Smaller batch sizes adds noise to gradient estimates
```

2. nn.Embedding -> Creates a dictioniory to map vectors to each vocab token  
   &emsp;Input:  
   &emsp;num_embeddings: dictionary size or vocab size  
   &emsp;embedding_dim : dim of vector (generally 512)  
   &emsp;Forward pass input:  
   &emsp;Tensor integer having vocab tokens ([batch size, sequence length])  
   &emsp;Forward pass Output:  
   &emsp;Tensor having embeddings of each token ([batch size, sequence length, embedding_dim])  
   &emsp;-> Output tensor is initially random and then trains as model is trained
3. nn.Dropout -> creates droptout layer
4. nn.Parameter -> Initialise traininable parameter
5. nn.Linear -> &emsp;W.x + b (W is trainable weight matrix and b is bias)
6. To compute M new tokens with a prompt length of N it will scale to $O(M \cdot (N+M)^2)$.  
   That's why we introduce KV Cache, old states are stored, dropping the incremental token generation step cost to $O(1)$ calculations per token relative to history length.
7. Storing the [B, H, S, S] attention matrix scores for the backward pass consumes massive amounts of memory, scaling quadratically ($O(S^2)$).  
   This is why optimizations like FlashAttention are introduced.

## Q1: How does Pre-LN and Post-LN in the FFN layer effect Gradient Flow and Learning Rate Sensitivity?

## 1. How Gradients Travel

### Post-LN (Original Paper)

$$x_{l+1} = \text{LN}(x_l + \text{Sub}(x_l))$$

Every gradient flowing backward **must pass through every LN Jacobian** in sequence.
The LN Jacobian is a projection matrix that:

- Zeroes out the component along the mean direction $[1, 1, ..., 1]$
- Zeroes out the component along the $(x - \mu)/\sigma$ direction (scale-invariance)
- Shrinks remaining components by $\leq 1$ (divison by $\sigma$)

Multiplying $N$ such matrices causes **exponential gradient shrinkage** with depth :
subspaces that survive one layer's projection may get killed by the next.

```
Loss → [J_LN] → [J_LN] → [J_LN] → ... → [J_LN] → Early layer weights
        Layer N   Layer N-1                 Layer 1
        (shrinks) (shrinks again)          (barely anything left)
```

### Pre-LN (Modern Standard)

$$x_{l+1} = x_l + \text{Sub}(\text{LN}(x_l))$$

The residual $x_l$ is added **outside** the LN, so the backward Jacobian is:
$$\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_{l+1}} \cdot \left(I + J_{\text{Sub}} \cdot J_{\text{LN}}\right)$$

The **identity term $I$** creates a direct gradient highway => a copy of the output
gradient flows back to every layer untouched, regardless of what the sublayer Jacobian does.

```
Loss
 │
 ├─────────────────────── Identity highway ──────────────────────────┐
 │                                                                   │
[Sub·LN Jacobian] → [Sub·LN Jacobian] → ... → [Sub·LN Jacobian]     │
                                                                     │
                                                       Early weights ←┘
```

Early layers are **guaranteed** to receive at least the magnitude of the output gradient.

---

## 2. Why Post-LN is Sensitive to Learning Rate

Because gradients shrink with depth in Post-LN, different layers have **wildly different
gradient magnitudes** at any given training step:

```
Layer 6:  ████████████████  (large gradient)
Layer 5:  ████████████
Layer 4:  ████████
Layer 3:  ████
Layer 2:  ██
Layer 1:  █                 (near-zero gradient)

          ↑ A single global LR must serve all of these
```

This creates an impossible trade-off:

| Learning Rate  | Late layers (5, 6)   | Early layers (1, 2)       |
| -------------- | -------------------- | ------------------------- |
| **Too high**   | Explode → divergence | Still okay                |
| **Too low**    | Learn slowly         | Learn essentially nothing |
| **Just right** | Marginal stability   | Still barely learning     |

There is **no single LR** that simultaneously keeps late layers stable and gives early
layers a meaningful signal. The warmup schedule in the original paper is a bandage:
start with near-zero LR (prevent late-layer explosion), then slowly increase (give
early layers a signal). But the fundamental imbalance remains throughout training.

Pre-LN avoids this entirely => the identity highway keeps gradient magnitudes roughly
**uniform across all layers**, so one LR works comfortably everywhere.

---

## Quick Reference

| Property                  | Post-LN (2017 Paper)      | Pre-LN (Modern)                        |
| ------------------------- | ------------------------- | -------------------------------------- |
| Gradient path             | Through every LN Jacobian | Identity highway exists                |
| Gradient magnitude (deep) | Shrinks exponentially     | Stays anchored to output               |
| LR sensitivity            | High (needs warmup)       | Low (stable at higher LR)              |
| Forward-pass activations  | Normalized per layer      | Residual stream grows → needs final LN |
| Used by                   | Original Transformer      | GPT-2/3, LLaMA, Mistral, Gemma         |

> **Note:** Post-LN's outputs _are_ normalized in the forward pass => the problem is
> purely in the **backward pass** (gradient flow), not in activation scales.

---

---

## Q2: Where is dropout applied? And how does modern approach change to applying dropout in FFN block?

### 1. After Positional Encoding

Applied on the sum of token embeddings + positional encodings.  
Forces the model not to over-rely on any specific position or embedding dimension
right from the start - regularizes the input before it enters the encoder/decoder stack.

> Paper (Section 5.4): _"we apply dropout to the sums of the embeddings and the positional encodings."_

---

### 2. On Attention Weights (inside Attention block)

Applied after softmax, before multiplying by V.  
After softmax, each row is a probability distribution over all tokens.
Without dropout, the model can memorize which tokens to always attend to
(**attention collapse**). Dropout randomly zeroes out weights, forcing the model
to spread attention and learn more robust, distributed patterns.

---

### 3. In Residual Connection block (On Sublayer Output)

```python
def forward(self, x, sublayer):
    return x + self.dropout(sublayer(self.norm(x)))
```

Applied on the output of each sublayer (Attention or FeedForward),
**before** adding back to the residual stream.  
The sublayer output is the new information being added to the residual stream  
Dropping parts of it forces the model not to rely too heavily on any single transformation

> Paper: _"We apply dropout to the output of each sub-layer, before it is added to the sub-layer input."_

---

### 4. Inside FeedForward - Between the Two Linear Layers (Modern Addition)

```python
def forward(self, x):
    x = F.relu(self.linear1(x))   # d_model → d_ff
    x = self.dropout(x)            # NOT in original paper
    x = self.linear2(x)            # d_ff → d_model
    return x
```

**Not in the original paper** - added in modern implementations.

The FFN expands to 4× the model size internally (`d_ff = 4 × d_model`).  
In this large intermediate space, neurons can **co-adapt** - always firing together
to produce memorized patterns.  
The Residual Connection's dropout acts on the final
compressed output and cannot prevent co-adaptation forming _inside_ the FFN.
Internal dropout breaks this by regularizing the expanded space directly.

---

## Full Dropout Flow

```
Token IDs
    │
[Text Encoding]               ← no dropout
    │
    + [Positional Encoding]
    │
[Dropout] ◄─────────────────── #1: input representation
    │
┌───────── Encoder Layer ──────────┐
│                                  │
│  [LayerNorm]                     │
│      │                           │
│  [Attention]                     │
│      │                           │
│  [Dropout] ◄─────────────────── #2: attention weights (inside Attention)
│      │                           │
│  [Dropout] ◄─────────────────── #3: sublayer output (inside Residual)
│      │                           │
│   x + sublayer(x)               │
│                                  │
│  (same structure for FFN layer)  │
│                                  │
│  [Dropout] ◄─────────────────── #4: inside FFN between linears (modern only)
│                                  │
└──────────────────────────────────┘
```

---

### When Is Internal FFN Dropout Actually Needed?

| Scenario                                  | Internal FFN Dropout Needed?                   |
| ----------------------------------------- | ---------------------------------------------- |
| Small model, small `d_ff`                 | Not much - residual dropout is enough          |
| Large model, large `d_ff`                 | Yes - co-adaptation risk grows with size       |
| Fine-tuning on small datasets (e.g. BERT) | Yes - overfitting risk is real                 |
| Large-scale pretraining (e.g. LLMs)       | Often removed - data volume is the regularizer |

At billion-token scale, the model rarely sees the same data twice,
so dropout isn't needed and only slows down convergence.

---

### Hyperparameter

All dropout locations share the **same `p` value**:

- Base model: `p = 0.1`
- Large model: `p = 0.3`

It is a single hyperparameter tuned across the entire architecture.

---

## Q3: Should EncoderBlock accept attention/FF layers as constructor arguments, or initialize them internally?

### Code in question

```python
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, ff_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.ff_block = ff_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x: Tensor, src_mask: Tensor = None):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.ff_block)
        return x
```

### Answer

**Accepting them as arguments (dependency injection) is the better approach**, and is the standard pattern used in research codebases like the Annotated Transformer and x-transformers.

**Advantages of dependency injection:**

- **Flexibility** :- you can pass in custom subclasses of `MultiHeadAttention` or `FeedForwardBlock` without modifying `EncoderBlock` at all. This is useful when experimenting with variants (e.g., linear attention, gated FFN).
- **Testability** :- you can inject mock or stub modules during unit testing.
- **Separation of concerns** :- `EncoderBlock` doesn't need to know _how_ to construct its components, only _how to use_ them.
- **Consistent with PyTorch idioms** :- this matches how `nn.TransformerEncoderLayer` and most serious implementations are structured.

The drawback of internal initialization is tight coupling :- swapping attention mechanisms would require subclassing or editing `EncoderBlock` itself.

---

## Q4: What is the difference between masking in cross-attention and self-attention?

### Context

For -
cross-attention: Padding mask  
self-attention: Causal/Lookahead mask

### Answer

The two masks serve completely different purposes.

---

#### Causal / Look-ahead Mask (Self-Attention)

Used in self-attention where `Q = K = V = x`.  
This is a **triangular mask** that prevents each position from attending
to future positions. It forces **autoregressive behaviour**.

---

#### Padding Mask (Cross-Attention)

Used in cross-attention where `Q = x` (decoder), `K = V = enc_out` (encoder).  
This masks out **padding tokens in the source sequence**. Sentences in a batch
have different lengths, so shorter ones get padded with dummy tokens to match the longest.  
The encoder should not attend to those padding positions - they
carry no real information.

---

#### Summary

|                    | `causal mask`                    | `padding mask`                  |
| ------------------ | -------------------------------- | ------------------------------- |
| Applied in         | Self-attention                   | Cross-attention                 |
| Masks              | Future target tokens             | Padding in source               |
| Shape              | `(d_model, d_model)` triangular  | `(1, src_len)` flat             |
| Purpose            | Prevent cheating during training | Ignore meaningless padding      |
| Changes per batch? | No (same triangle always)        | Yes (depends on source lengths) |

## Q5: What type of normalization do we use and why ?

## Q6: Why exactly do we divide the query-key dot product by $\sqrt{d_k}$? What happens to the gradients during backpropagation if you omit this?

## Q7: What is the significance of warm-up steps?
