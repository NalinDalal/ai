# Toy English→French Translator (Seq2Seq)

A simple yet complete implementation of a neural machine translation model using the Sequence-to-Sequence (Seq2Seq) architecture with LSTM encoder-decoder.

---

## The Problem

Machine translation—automatically converting text from one language to another—has been a long-standing challenge in natural language processing. Traditional approaches relied on rule-based systems or statistical models that required extensive hand-crafted features and parallel corpora.

With the rise of deep learning, **neural machine translation (NMT)** emerged as a paradigm shift. Instead of building complex pipelines of alignment, phrase tables, and language models, NMT trains a single end-to-end neural network that directly maps source text to target text.

The key challenge: **how do we handle sequences of variable length where input and output lengths differ?**

---

## Our Approach: Seq2Seq with LSTM Encoder-Decoder

We implemented a model based on the landmark paper **"Sequence to Sequence Learning with Neural Networks"** by Sutskever, Vinyals & Le (2014). The core idea is elegantly simple:

1. **Encoder**: An LSTM reads the source sentence (English) one token at a time and compresses it into a fixed-dimensional **context vector** (the final hidden state).

2. **Decoder**: Another LSTM takes the context vector and generates the target sentence (French) one token at a time, using the previously generated token as input for the next step.

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
- Output: Final hidden state and cell state (context vector)

**Decoder (Decoder class)**
- Embedding layer: Similar to encoder
- Multi-layer LSTM: Generates next token conditioned on context
- Linear layer: Projects hidden state to vocabulary size

**Seq2Seq (Seq2Seq class)**
- Combines encoder and decoder
- Implements teacher forcing during training
- Implements greedy decoding during inference

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
├── Vocabulary           # Token-to-index and index-to-token mapping
├── Encoder              # LSTM encoder for source language
├── Decoder              # LSTM decoder for target language
├── Seq2Seq              # Complete encoder-decoder model
├── TranslationDataset   # PyTorch Dataset for translation pairs
├── get_toy_dataset()    # Sample English-French pairs
├── train_model()        # Training loop
└── translate_sentence() # Inference function
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

### Training the Model

```bash
python eng2Fre.py --train --epochs 100
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

| Input (English) | Output (French) |
|-----------------|-----------------|
| hello | bonjour |
| thank you | merci |
| goodbye | au revoir |
| how are you | comment allez vous |
| i love you | je t'aime |

---

## Limitations & Future Improvements

### Current Limitations
1. **Small vocabulary**: Only handles words seen during training
2. **Tiny dataset**: 30 pairs is insufficient for real-world translation
3. **No attention**: Basic Seq2Seq struggles with long sentences
4. **Greedy decoding**: Doesn't explore multiple translation candidates

### Possible Improvements
1. **Attention Mechanism** (Bahdanau et al., 2014): Let decoder focus on relevant source words
2. **Larger Dataset**: Use WMT14 or similar benchmark datasets
3. **Beam Search**: Explore multiple translation paths instead of greedy selection
4. **Bidirectional Encoder**: Process source sentence in both directions
5. **Transformer Architecture**: Replace LSTM with self-attention (Vaswani et al., 2017)

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