# AI

My AI engineering learning index.

This repository does **not** contain implementations. It is the public map of what I'm learning, what I've built, and where the actual work lives.

---

## Currently

**Learning:** Python fundamentals, NumPy, linear algebra, probability, statistics

**Building:** [machine-learning](https://github.com/NalinDalal/machine-learning) — implementing classical ML algorithms from scratch

**Reading:** See [ai.md](./ai.md) for the full curriculum roadmap

**Current question:** How does optimization actually behave when we change batch size and learning rate?

---

## Approach

I'm approaching this as an engineering apprenticeship:

> **learn → investigate → build → measure → break → understand → publish**

This repo is the index. The actual implementations, experiments, and notes live in standalone repositories linked below. The goal is a visible body of work that shows how I think, debug, and reason about real engineering problems — not a portfolio of completed tutorials.

---

## Repositories

### Foundations

| Topic | Repository | Description |
|-------|-----------|-------------|
| Python + Math | [python-basics](https://github.com/NalinDalal/python-basics) | Python, NumPy, Pandas, matplotlib, linear algebra, calculus, statistics, probability |
| Classical ML | [machine-learning](https://github.com/NalinDalal/machine-learning) | Linear models, trees, SVMs, clustering, PCA, autoencoders, RL basics |
| Neural Networks | [neural-networks](https://github.com/NalinDalal/neural-networks) | Perceptron, backpropagation, DLS course mapping, paper notes |
| From-scratch NN library | [own-neural-net](https://github.com/NalinDalal/own-neural-net) | Dense layers, activations, optimizers, losses, model training — no high-level APIs |
| Autograd reference | [micrograd](https://github.com/NalinDalal/micrograd) | Karpathy's tiny autograd engine and NN library |

### Deep Learning

| Topic | Repository | Description |
|-------|-----------|-------------|
| CNNs | [cnn-architectures](https://github.com/NalinDalal/cnn-architectures) | NumPy + PyTorch CNN comparison, CIFAR-10, ResNet, face recognition |
| AlexNet paper | [ImageNet-Classification-with-Deep-Convolutional-Neural-Networks-AlexNet-paper](https://github.com/NalinDalal/ImageNet-Classification-with-Deep-Convolutional-Neural-Networks-AlexNet-paper) | Paper notes and implementation |
| Object Detection (YOLO) | [object-detect-cnn](https://github.com/NalinDalal/object-detect-cnn) | YOLO paper notes and implementation |
| Deep Learning | [week-9-deep-learning](https://github.com/NalinDalal/week-9-deep-learning) | Backprop, initialization, regularization, optimization |
| RNNs | [rnn](https://github.com/NalinDalal/rnn) | RNN, LSTM from scratch with BPTT, gradient clipping |
| RNNs (Karpathy) | [rnn-karpathy](https://github.com/NalinDalal/rnn-karpathy) | Character-level LSTM, truncated BPTT, temperature sampling |
| LSTM + Word2Vec | [lstm-word2Vec](https://github.com/NalinDalal/lstm-word2Vec) | LSTM language modeling, Word2Vec, t-SNE visualizations |
| Sequence Models | [Sequence-Models](https://github.com/NalinDalal/Sequence-Models) | Seq2Seq, NMT, attention, English→French translator |
| Transformers | [transformers](https://github.com/NalinDalal/transformers) | Transformer architecture, ViT, BPE tokenization, Switch Transformer |

### Research

| Topic | Repository | Description |
|-------|-----------|-------------|
| Paper Reproductions | See individual repos above | Experiments and reproductions are kept alongside implementations |
| Research Notes | See [ai.md](./ai.md) | Complete curriculum roadmap, paper list, and resource index |

---

## How to Use This Repo

- **Start here:** Read this README to understand what is being built and where.
- **Navigate to implementations:** Each row in the tables above links to a standalone repository with actual code, READMEs, and experiment logs.
- **Curriculum reference:** [ai.md](./ai.md) contains the complete learning roadmap, paper list, and resource index.
- **Project ideas:** [project.md](./project.md) contains project briefs and future plans.
- **Reading tracker:** [paper-read.md](./paper-read.md) tracks papers being read.

---

## What This Repo Does Not Contain

- No implementation files (`.py`, `.ipynb`, etc.)
- No experiment outputs or model checkpoints
- No generated artifacts

Those belong in the repositories linked above.

---

## Structure

```
NalinDalal/ai          ← You are here. Index only.
├── README.md          ← This file. The map.
├── ai.md              ← Complete curriculum, papers, resources
├── project.md         ← Project ideas and briefs
├── paper-read.md      ← Paper reading tracker
└── research-notes/    ← Learning notes and summaries
```
