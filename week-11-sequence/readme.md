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

## Papers

Day 4(03.04):
- 11. [**Bahdanau, Cho & Bengio (2014)** - "Neural Machine Translation by Jointly Learning to Align and Translate"](https://arxiv.org/pdf/1409.0473)
- [yt-link](https://youtu.be/XXtpJxZBa2c?si=oc6fOm5VPPCYFauY)


day 5(04.04):
- **Mini-project:** Add attention to your seq2seq
