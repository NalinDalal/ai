# Toy English→French Translator (Seq2Seq with Attention)

A simple yet complete implementation of a neural machine translation model using the Sequence-to-Sequence (Seq2Seq) architecture with LSTM encoder-decoder and **Bahdanau Attention**.

---

## The Problem

Machine translation—automatically converting text from one language to another—has been a long-standing challenge in natural language processing. Traditional approaches relied on rule-based systems or statistical models that required extensive hand-crafted features and parallel corpora.

With the rise of deep learning, **neural machine translation (NMT)** emerged as a paradigm shift. Instead of building complex pipelines of alignment, phrase tables, and language models, NMT trains a single end-to-end neural network that directly maps source text to target text.

The key challenge: **how do we handle sequences of variable length where input and output lengths differ?**

---

## Our Approach: Seq2Seq with LSTM Encoder-Decoder + Attention

We implemented a model based on the landmark papers:
- **"Sequence to Sequence Learning with Neural Networks"** by Sutskever, Vinyals & Le (2014)
- **"Neural Machine Translation by Jointly Learning to Align and Translate"** by Bahdanau, Cho & Bengio (2014)

The core idea combines encoder-decoder architecture with an **attention mechanism** that allows the decoder to focus on relevant parts of the input at each step.

1. **Encoder**: An LSTM reads the source sentence (English) one token at a time and produces **all hidden states** (not just the final one).

2. **Attention Mechanism**: At each decoding step, the decoder computes attention weights over all encoder hidden states to determine which source words are most relevant.

3. **Decoder**: Another LSTM generates the target sentence (French) one token at a time, using both the previous token AND the attention-weighted context vector.

### Architecture Overview (with Attention)

```
English: "how are you" 
         ↓
┌─────────────────────────────────────────────────────────────────┐
│                        ENCODER                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                │
│  │Embedding │ →  │ BiLSTM   │ →  │ All Hiddens │ ──────────→ │
│  │  Layer   │    │          │    │ (for attention) │        │
│  └──────────┘    └──────────┘    └──────────┘                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌───────────────────┐
                    │  ATTENTION         │
                    │  (Bahdanau)       │
                    │  scores → softmax  │
                    └───────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        DECODER                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐ │
│  │  <SOS>   │ →  │   LSTM   │ →  │  Linear  │ →  │ softmax│ │
│  │+context  │    │+context  │    │          │    │        │ │
│  └──────────┘    └──────────┘    └──────────┘    └────────┘ │
│        ↑                                                            │
│        └────────── (previous output feeds back) ──────────       │
└─────────────────────────────────────────────────────────────────┘
```

### How Attention Works

The attention mechanism computes a **context vector** at each decoder step:

```
1. Attention Scores: e_t = v^T · tanh(W_a · s_t + U_a · h_i)
2. Attention Weights: α_t = softmax(e_t)
3. Context Vector:    c_t = Σ α_t[i] · h_i

Where:
  - s_t = decoder hidden state at time t (query)
  - h_i = encoder hidden state at time i (key/value)
  - α_t[i] = attention weight for encoder state i (how much to "look at" position i)
```

### Architecture Overview

```
English: "hello" → [Encoder LSTM] → [Context Vector] → [Decoder LSTM] → "bonjour"
```

```
┌─────────────────────────────────────────────────────────────────┐
│                         ENCODER                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                 │
│  │Embedding │ →  │   LSTM   │ →  │  Context │                 │
│  │  Layer   │    │  (x n)  │    │  Vector  │                 │
│  └──────────┘    └──────────┘    └──────────┘                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         DECODER                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐  │
│  │   <SOS>  │ →  │   LSTM   │ →  │  Linear  │ →  │ softmax │  │
│  └──────────┘    └──────────┘    └──────────┘    └────────┘  │
│        ↑                                                            │
│        └────────── (previous output feeds back) ──────────       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Details

### Vocabulary

We use a simple token-based vocabulary with special tokens:

| Token | Index | Purpose |
|-------|-------|---------|
| `<PAD>` | 0 | Padding for fixed-length sequences |
| `<SOS>` | 1 | Start-of-sequence (decoder input start) |
| `<EOS>` | 2 | End-of-sequence (marks translation end) |
| `<UNK>` | 3 | Unknown token for out-of-vocabulary words |

### Model Components

**Encoder (Encoder class)**
- Embedding layer: Maps token indices to dense vectors (embed_dim)
- Multi-layer LSTM: Processes the embedded sequence
- Output: 
  - Final hidden state and cell state (context vector for initialization)
  - **ALL encoder hidden states** (used by attention mechanism)

**Decoder (Decoder class - basic)**
- Embedding layer: Similar to encoder
- Multi-layer LSTM: Generates next token conditioned on context
- Linear layer: Projects hidden state to vocabulary size

**DecoderAttention (Decoder class - with Attention)**
- Embedding layer: Maps tokens to dense vectors
- **Attention layers** (Bahdanau):
  - `W_a`: Linear layer to transform decoder state
  - `U_a`: Linear layer to transform encoder states  
  - `v_a`: Linear layer to compute scalar scores
- LSTM: Takes embedding + context vector as input
- Linear layer: Projects to vocabulary size

**Key difference**: With attention, the decoder doesn't rely on a single fixed context vector. Instead, it computes a **new context vector at each decoding step** by weighting all encoder hidden states.

**Seq2Seq (Seq2Seq class)**
- Combines encoder and decoder
- Supports **optional attention** (enabled by default)
- Implements teacher forcing during training
- Implements greedy decoding during inference

### Attention Mechanism Details

We use **Bahdanau Attention** (also called "additive attention"):

```python
# Compute attention scores
scores = v_a(T.tanh(W_a * s_t + U_a * h_i))

# Softmax to get weights
attn_weights = softmax(scores)

# Weighted sum of encoder hidden states
context = sum(attn_weights[i] * h_i for all i)
```

The decoder input becomes: `[embedded_token; context_vector]` instead of just `[embedded_token]`

This allows the model to:
- **Handle long sentences** better (doesn't rely on single context vector)
- **Learn alignments** automatically (which source words map to which target words)
- **Improve translation quality** especially for sentences where word order differs

### Training

- **Loss function**: Cross-Entropy Loss (ignores padding)
- **Optimizer**: Adam (learning rate = 0.001)
- **Teacher forcing**: 50% probability—during training, sometimes we feed the ground-truth previous token instead of the model's prediction
- **Gradient clipping**: Threshold = 1.0 to prevent exploding gradients

### Inference

- **Greedy decoding**: At each step, select the token with highest probability
- **Stop condition**: When `<EOS>` token is generated or max length reached

---

## Code Structure

```
eng2Fre.py
├── Vocabulary              # Token-to-index and index-to-token mapping
├── Encoder                 # LSTM encoder for source language (outputs all hidden states)
├── Decoder                 # Basic LSTM decoder (no attention)
├── DecoderAttention        # LSTM decoder with Bahdanau attention
├── Seq2Seq                 # Complete encoder-decoder model (supports attention)
├── TranslationDataset      # PyTorch Dataset for translation pairs
├── get_toy_dataset()       # Sample English-French pairs
├── train_model()           # Training loop
└── translate_sentence()    # Inference function
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| embed_dim | 128 | Dimensionality of word embeddings |
| hidden_dim | 256 | LSTM hidden state size |
| num_layers | 2 | Number of LSTM layers |
| dropout | 0.2 | Dropout probability |
| batch_size | 4 | Training batch size |
| epochs | 100 | Number of training epochs |

---

## Usage

### Training the Model (with Attention)

```bash
python eng2Fre.py --train --epochs 100
```

### Training without Attention (basic Seq2Seq)

```bash
python eng2Fre.py --train --epochs 100 --no-attention
```

### Translating a Sentence

```bash
python eng2Fre.py --translate "hello"
```

### Running Demo

```bash
python eng2Fre.py --demo
```

### Custom Parameters

```bash
python eng2Fre.py --train --embed_dim 256 --hidden_dim 512 --epochs 200
```

---

## Results

With our toy dataset (30 English-French pairs), the model learns basic translations after training:

### With Attention (default)

| Input (English) | Output (French) |
|-----------------|-----------------|
| hello | bonjour |
| thank you | merci |
| goodbye | au revoir |
| how are you | comment allez vous |
| i love you | je t'aime |

### Without Attention (--no-attention flag)

| Input (English) | Output (French) |
|-----------------|-----------------|
| hello | bonjour |
| thank you | merci |
| goodbye | au revoir |
| how are you | comment allez |
| i love you | je t'aime |

Note: With attention, the model can better handle longer sentences and maintain better word alignment.

---

## Attention vs Basic Seq2Seq: Key Differences

| Aspect | Basic Seq2Seq | Seq2Seq + Attention |
|--------|---------------|---------------------|
| Context | Single fixed vector | Dynamic context at each step |
| Long sentences | Struggles | Handles better |
| Alignment | Not explicit | Learned implicitly |
| Parameters | Fewer | +3 linear layers |
| Memory | Lower | Slightly higher |

### Why Attention Matters

In basic Seq2Seq, the encoder compresses the entire source sentence into a single context vector. This creates a bottleneck—especially for long sentences where the decoder must reconstruct many words from limited information.

Attention fixes this by:
1. **Keeping all encoder hidden states** instead of just the final one
2. **Computing a weighted context** at each decoder step
3. **Letting the model decide** which source words are relevant for each target word

---

## Limitations & Future Improvements

### Current Limitations
1. **Small vocabulary**: Only handles words seen during training
2. **Tiny dataset**: 30 pairs is insufficient for real-world translation
3. **Greedy decoding**: Doesn't explore multiple translation candidates
4. **Single direction encoder**: Could use bidirectional LSTM

### Possible Improvements
1. **Beam Search**: Explore multiple translation paths instead of greedy selection
2. **Bidirectional Encoder**: Process source sentence in both directions for better context
3. **Luong Attention**: Try multiplicative attention (often faster)
4. **Transformer Architecture**: Replace LSTM with self-attention (Vaswani et al., 2017)
5. **Larger Dataset**: Use WMT14 or similar benchmark datasets
6. **Subword Tokenization**: Use BPE/WordPiece for handling OOV words

---

## References

1. **Sutskever, Vinyals & Le (2014)** - "Sequence to Sequence Learning with Neural Networks" - [arXiv:1409.3215](https://arxiv.org/abs/1409.3215)

2. **Bahdanau, Cho & Bengio (2014)** - "Neural Machine Translation by Jointly Learning to Align and Translate" - [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)

3. **Vaswani et al. (2017)** - "Attention Is All You Need" - [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

4. **English to French Translation using Seq2Seq (Jupyter Notebook)** - [GitHub: SayamAlt/English-to-French-Language-Translation-using-Seq2Seq-Modeling](https://github.com/SayamAlt/English-to-French-Language-Translation-using-Seq2Seq-Modeling)

---

## Appendix: Complete Training Output Example

```
Device: cpu

Loading data...
Source vocabulary: 30 tokens
Target vocabulary: 52 tokens

Training for 10 epochs...
Epoch  10: Loss = 0.8234

Demo translations:
  hello -> bonjour
  thank you -> merci
  goodbye -> au revoir
  how are you -> comment allez
  i love you -> je t'aime
```