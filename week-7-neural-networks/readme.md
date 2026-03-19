# 03.03.2026 to 08.03.2026

## Daily Schedule (~3-4 hrs/day)

| Date | Day | Topics | Tasks |
|------|-----|--------|-------|
| 06.03 (Fri) | 6 | Backprop Paper (Steps 5-8) + Mini-project | Single-hidden-layer NN in NumPy; gradient check; write 300-500 word blog |
| 07.03 (Sat) | 7 | **DLS C1 W4:** Deep Neural Networks | Building Deep NN Step-by-Step notebook → DNN Image Classification notebook; W4 quiz |
| 08.03 (Sun) | 8 | Review + nnfs.io start | Consolidate Week 7; push all code + blogs; begin nnfs.io Ch. 1 |

---

## Week 7: Neural Network Fundamentals + Stage A Papers

### Stage A — Roots & Learning Fundamentals (EXACT ORDER)

#### Paper 1 — McCulloch & Pitts (1943) - Done
[neurons as logic gates](./neuron-as-logic-gate/readme.md)

#### Paper 2 — Rosenblatt (1958) - Done
[Perceptron](./perceptron/readme.md)

---

#### Paper 3 — Rumelhart, Hinton & Williams (1986)

[Backpropagation](./backpropogation/readme.md)

**Split across Day 5 (Steps 1-4) and Day 6 (Steps 5-8):**

**Day 5 — after the W3 notebook:**
- [ ] **Step 1 — Skim (10-15m):** Title, abstract, intro, conclusion, figures. Write one-sentence claim.
- [ ] **Step 2 — Questions (5m):** What problem is it solving? Intended goal? What makes it hard?
- [ ] **Step 3 — Deep read (30-60m):** Chain rule derivation + delta rule; rederive the weight update equation by hand.
- [ ] **Step 4 — Code hunt (15-30m):** Find a clean NumPy backprop reference implementation.

> You've just seen a shallow NN run in the W3 notebook. Now you know the math behind *why* it works.

**Day 6 — implement + close out:**
- [ ] **Step 5 — Implement:** Single-hidden-layer NN in NumPy (tanh hidden, sigmoid output); gradient check to verify derivatives. *(Revisited Week 9 — full treatment: initialization, regularization, gradient checking as standalone topics in DLS C2 W1)*
- [ ] **Step 6 — Result:** What actually happened, trade-offs, what worked vs didn't.
- [ ] **Step 7 — Strength + Improvement:** Why powerful, where it breaks, what you'd improve.
- [ ] **Step 8 — Blog (20-30m):** 300-500 words — problem / method / main result / limitations / one-line extension idea.

> `Rule:` If you can't implement a minimal version in 2 days, you didn't read it properly.

---

### DLS Course 1 — Neural Networks and Deep Learning

> Do these **in order**, one per day. Each day's course material directly prepares you for the next day's paper or project.

#### Day 5 → DLS C1 Week 3 + Backprop Paper (Steps 1-4)

| # | Assignment | What You Learn | Note |
|---|-----------|----------------|------|
| 2 | [Planar Data Classification with One Hidden Layer](https://nbviewer.jupyter.org/github/amanchadha/coursera-deep-learning-specialization/blob/master/C1%20-%20Neural%20Networks%20and%20Deep%20Learning/Week%203/Planar%20data%20classification%20with%20one%20hidden%20layer/Planar_data_classification_with_onehidden_layer_v6c.ipynb) | Hidden layers, non-linearity, tanh — the exact architecture you'll build from scratch on Day 6 | Regularization + dropout *Revisited Week 9 — DLS C2 W1* |

**Quiz:** [W3 — Shallow Neural Networks](https://nbviewer.jupyter.org/github/amanchadha/coursera-deep-learning-specialization/blob/master/C1%20-%20Neural%20Networks%20and%20Deep%20Learning/Week%203/Week%203%20Quiz%20-%20Shallow%20Neural%20Networks.md)

#### Day 6 → Backprop Mini-Project (Paper Steps 5-8)

Build what Day 5 taught — no scaffolding, just NumPy:

- Single-hidden-layer NN (tanh hidden, sigmoid output)
- Forward pass → compute loss → backward pass via chain rule
- Gradient check — confirm derivatives are correct *(Revisited Week 9 — standalone DLS C2 W1 notebook)*
- Write the 300-500 word blog entry
- Push everything to repo

#### Day 7 → DLS C1 Week 4: Deep Neural Networks

| # | Assignment | What You Learn | Note |
|---|-----------|----------------|------|
| 3 | [Building your Deep NN: Step by Step](https://nbviewer.jupyter.org/github/amanchadha/coursera-deep-learning-specialization/blob/master/C1%20-%20Neural%20Networks%20and%20Deep%20Learning/Week%204/Building%20your%20Deep%20Neural%20Network%20-%20Step%20by%20Step/Building_your_Deep_Neural_Network_Step_by_Step_v8a.ipynb) | L-layer forward/backward, parameter init, cache — generalises Day 6's 2-layer net | Weight init strategies *Revisited Week 9 — DLS C2 W1 (zero vs random vs He)* |
| 4 | [Deep NN for Image Classification: Application](https://nbviewer.jupyter.org/github/amanchadha/coursera-deep-learning-specialization/blob/master/C1%20-%20Neural%20Networks%20and%20Deep%20Learning/Week%204/Deep%20Neural%20Network%20Application_%20Image%20Classification/Deep%20Neural%20Network%20-%20Application%20v8.ipynb) | End-to-end cat classifier — your first real DL model | PyTorch + framework transition *Revisited Week 8 — first PyTorch use on MNIST/CIFAR-10* · TensorFlow *Revisited Week 9 — DLS C2 W3* |

**Quiz:** [W4 — Key Concepts on Deep Neural Networks](https://nbviewer.jupyter.org/github/amanchadha/coursera-deep-learning-specialization/blob/master/C1%20-%20Neural%20Networks%20and%20Deep%20Learning/Week%204/Week%204%20Quiz%20-%20Key%20concepts%20on%20Deep%20Neural%20Networks.md)

---

### How the Pieces Connect (Linear Path)

```
McCulloch & Pitts (1943)     → neurons are logic gates
        ↓
Rosenblatt (1958)            → learns weights from data (perceptron)
        ↓
DLS C1 W1-W2                 → same idea formalized: logistic regression = 1-neuron net
        ↓
DLS C1 W3                    → add a hidden layer → shallow NN (uses backprop)
        ↓
Rumelhart et al. (1986)      → the math behind why that hidden layer can train at all
        ↓
Backprop mini-project        → build it from scratch in NumPy, verify gradients
        ↓
DLS C1 W4                    → generalize to L layers → deep neural network
        ↓
700-page nnfs.io             → rebuild everything from first principles, no frameworks
```

---

### What Week 7 Intentionally Leaves Shallow

> These topics appear this week but are treated as *exposure only*. Full implementation comes later.

| Topic | First Seen | Deep Treatment |
|-------|-----------|----------------|
| Gradient checking | Day 6 mini-project | Week 9 — DLS C2 W1 standalone notebook |
| Weight initialization (zero vs random vs He) | DLS C1 W4 | Week 9 — DLS C2 W1 |
| Regularization + dropout | DLS C1 W3 | Week 9 — DLS C2 W1 |
| Optimization (SGD, momentum, Adam) | DLS C1 W2 | Week 9 — DLS C2 W2 |
| PyTorch | — | Week 8 — first use on MNIST/CIFAR-10 |
| TensorFlow | — | Week 9 — DLS C2 W3 |

---

### Where the Rest of DLS Goes (Future Weeks)

| DLS Course | Maps To | Your Week |
|------------|---------|-----------|
| **C1:** Neural Networks and Deep Learning | Stage A papers + NN fundamentals | **Week 7** (this week) |
| **C2:** Improving Deep NNs (Hyperparams, Regularization, Optimization) | Deep Learning Mastery | **Week 9** |
| **C3:** Structuring ML Projects | ML strategy (quizzes only) | **Week 9** |
| **C4:** Convolutional Neural Networks | Stage B papers (LeCun, AlexNet, ResNet) | **Week 8** |
| **C5:** Sequence Models | Stage C papers (LSTM, Seq2Seq, Attention) | **Week 10-11** |

---

### Additional Resources

- [DLS Full Repository](https://github.com/amanchadha/coursera-deep-learning-specialization/tree/master)
- [DLS C1 Course Notes](https://github.com/amanchadha/coursera-deep-learning-specialization/tree/master/C1%20-%20Neural%20Networks%20and%20Deep%20Learning/Notes)
- [Neural Networks from Scratch — nnfs.io](https://nnfs.io/) (700-page guide, begin Day 8)