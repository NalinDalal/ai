# [**Sutskever, Vinyals & Le (2014)** — Seq2Seq](https://arxiv.org/pdf/1409.3215)

## introduction
DNN(Deep Neural Networks) are way too powerful
ex: sort N N-bit number using only 2 hidden layers of quadratic size
can be trained with backpropogation if we have large enough dataset

However, traditional DNNs assume fixed-size input and output vectors, making them unsuitable for sequence problems such as translation where sentence lengths vary.


**The Solution: Seq2Seq Models**

challenge: dimension of input and output must be known and fixed
solution: LSTM
The idea is to use one LSTM to read the input sequence, one timestep at a time, to obtain large fixed dimensional vector representation, and then to use another LSTM to extract the output sequence from that vector

2nd LSTM: RNN model conditioned on input sequence

considerable choice cause considerable time lag between the inputs and their corresponding outputs

various attempts have been made to address Seq2Seq model with neural networks.
map the entire input sentence to vector

another attempt say to allows neural networks to focus on different parts of their inputs

applied to English to French Translation task, a BLEU score of 34.81 by directly extracting translations from an ensemble of 5 deep
LSTMs (with 384M parameters and 8,000 dimensional state each) using a simple left-to-right beamsearch decoder.

used the LSTM to rescore the publicly available 1000-best lists of the SMT baseline on the same task. BLEU: 36.5 

able to do well on long sentences because we reversed the order of words in the source sentence but not the target sentences in the training and test set. 
By doing so, we introduced many short term dependencies that made the optimization problem much simpler.

LSTM learns to map an input sentence of variable length into a fixed-dimensional vector representation.

## Model
RNN is generalisation of feedforward neural networks to sequence.
Given a sequence of inputs (x1, . . . , xT ), a standard RNN computes a
sequence of outputs (y1, . . . , yT ) by iterating the following equation:

$$h_t = \tanh(W^{hx}x_t + W^{hh}h_{t-1} + b_h)$$
$$y_t = W^{yh}h_t + b_y$$

simplest strategy is to map the input sequence to a fixed-sized vector using one RNN, and then to map the vector to the target sequence with another RNN.

use LSTM to train this RNN
goal of LSTM is to estimate conditional probability p(y1, . . . , yT′ |x1, . . . , xT ) where
(x1, . . . , xT ) is an input sequence and y1, . . . , yT′ is its corresponding output sequence whose length T′ may differ from T .

$$p(y_1, ..., y_{T'} | x_1, ..., x_T) = \prod_{t=1}^{T'} p(y_t | v, y_1, ..., y_{t-1})$$

distribution is represented by softmax over words in vocab
LSTM computes representation of “A”, “B”, “C”, “<EOS>” and then uses this representation to compute the probability of “W”, “X”, “Y”, “Z”, “<EOS>”.

different in 3 ways:
1. Used 2 instead of 1(input sequence, output sequence)
2. Deep LSTM>Shallow LSTM; used 4 layer LSTM
3. it extremely valuable to reverse the order of the words of the input sentence.

instead of mapping a,b,c to α, β, γ, LSTM asks to map c, b, a to α, β, γ, where α, β, γ is the translation of a, b, c

## Experiments
2 ways: 
1. Direct translation using Seq2Seq model.
2. Rescoring 1000-best outputs from an SMT baseline.


**DataSet**: 
- 12 million sentence pairs
- 348M French words
- 304M English words
- Fixed vocabulary representation for each word

fixed vocabulary for both languages.

---

**Decoding & Rescoring**: 
train by maximising the log probability of a correct translation T given the source sentence S, training objective:
$$\frac{1}{|S|} \sum_{(T,S) \in S} \log p(T|S)$$

S = training set

produce translations by finding the most likely translation :
$$\hat{T} = \arg\max_T p(T|S)$$

search for the most likely translation using a simple left-to-right beam search decoder which
maintains a small number B of partial hypotheses, where a partial hypothesis is a prefix of some
translation. 
At each timestep we extend each partial hypothesis in the beam with every possible word in the vocabulary. 
This greatly increases the number of the hypotheses so we discard all but the B most likely hypotheses according to the model’s log probability.

decoder is approximate; system works well with beam size=1; beam size=2 provides most of the benefits of beam search

**Reversing the Source Sentences**

Reversing source sentences significantly improved results: perplexity dropped from 5.8 to 4.7, and BLEU increased from 25.9 to 30.6.

**Why it works:**
When concatenating source and target, each source word is far from its corresponding target word, creating a large "minimal time lag." Reversing makes the first few words of source close to the first few words of target, greatly reducing this lag and making optimization easier.

Contrary to the belief that LSTM would be more confident in early target parts, reversed source consistently outperformed raw source.

---

## Training Setup

1. **Model setup:** 4-layer deep LSTM with 1000 cells per layer, 1000-dim word embeddings, vocab size 160k (input) and 80k (output), ~384M parameters.
2. **Optimization:** Stochastic Gradient Descent (no momentum), learning rate 0.7; after 5 epochs, learning rate halved every 0.5 epoch; trained for 7.5 epochs.
3. **Batching:** Batch size = 128 sequences; gradients averaged by dividing by 128.
4. **Stability & efficiency:** Gradient clipping applied if norm > 5 to prevent exploding gradients; minibatches grouped by similar sentence lengths for ~2× faster training.


1. **Architecture distribution:** Model uses 8 GPUs; 4 GPUs run the 4 LSTM layers (one layer per GPU).
2. **Parallel softmax:** Remaining 4 GPUs split the softmax computation across vocabulary outputs.
3. **Computation split:** Each softmax GPU handles multiplication of large matrices (~1000 × 20000).
4. **Performance gain:** Parallelization significantly improves training speed and efficiency.


| Aspect | Details |
|--------|---------|
| Architecture | 4-layer deep LSTM, 1000 cells/layer |
| Embeddings | 1000-dim word embeddings |
| Vocabulary | 160k (input), 80k (output) |
| Parameters | ~384M |
| Batch size | 128 sequences |
| Optimization | SGD (no momentum), lr=0.7, halved every 0.5 epoch after epoch 5 |
| Stability | Gradient clipping (norm > 5) |
| Speedup | Grouped similar-length sentences (~2× faster) |
| Hardware | 8 GPUs (4 for LSTM layers, 4 for parallel softmax) |

---

## Experiment Results

**Direct Translation:**
| Method | BLEU |
|--------|------|
| Single forward LSTM, beam=12 | 26.17 |
| Single reversed LSTM, beam=12 | 30.59 |
| Ensemble 5 LSTMs, beam=2 | 34.50 |
| Ensemble 5 LSTMs, beam=12 | **34.81** |

**As SMT Rescorer:**
| Method | BLEU |
|--------|------|
| Baseline SMT | 33.30 |
| Single reversed LSTM | 35.85 |
| Ensemble 5 LSTMs | **36.5** |
| Oracle (upper bound) | ~45 |

**Key Finding:** LSTM handles long sentences better, especially with reversed source input.


## Conclusion

The Seq2Seq LSTM model demonstrates that:

1. **End-to-end learning** can match and exceed traditional SMT systems without manual feature engineering
2. **Simple techniques** (reversing input, beam search, ensembling) yield significant improvements
3. **Deep LSTMs** (4 layers, 1000 units) effectively compress variable-length sequences into fixed representations
4. **Hybrid approach** of using LSTM to rescore SMT outputs provides complementary benefits

This work established the encoder-decoder architecture as a foundation for neural machine translation, later improved by attention mechanisms.
