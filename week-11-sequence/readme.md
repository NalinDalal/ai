# Week 11: Sequence Models

[**Distributed Representations of Words and Phrases and their Compositionality**](./DistribuRepreWordPhrase.md) 

[**Sutskever, Vinyals & Le (2014)** — Seq2Seq](./Seq2Seq.md)

## Implementation

| File | Description |
|------|-------------|
| `seq2seq.py` | NumPy implementation from scratch |
| `seq2seq_pytorch.py` | PyTorch implementation (recommended) |
| `data_utils.py` | Vocabulary, data loading, batching |
| `train.py` | Training script |

### Run Training
```bash
cd week-11-sequence
python train.py          # NumPy version
python seq2seq_pytorch.py # PyTorch version
```

**Mini-project:** Toy English→French translator (seq2seq)
[Eng2Fre](./Eng2Fre.md)

[Neural Machine Translation by Jointly Learning to Align and Translate](./NMT.md)


day 5(04.04):
- **Mini-project:** Add attention to your seq2seq
