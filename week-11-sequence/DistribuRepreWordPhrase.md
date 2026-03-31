# [Distributed Representations of Words and Phrases and their Compositionality- Mikolov et al. (2013)](https://arxiv.org/pdf/1310.4546)

---

## Introduction

Distributed representations of words in a vector space help learning algorithms achieve better performance in natural language processing tasks by grouping similar words.

**Skip-gram Model** represents words as vectors learned from unstructured text without dense matrix multiplications. The learned vectors explicitly encode many linguistic regularities and patterns.

### Key Insight

> vec("Madrid") - vec("Spain") + vec("France") ≈ vec("Paris")

### The Problem with Word Representations

Word representations are limited by their inability to represent idiomatic phrases that are not compositions of individual words. For example, "Boston Globe" is a newspaper, not a natural combination of "Boston" and "Globe".

### Vector Addition is Powerful

Simple vector addition can often produce meaningful results:

- vec("Russia") + vec("river") ≈ vec("Volga River")
- vec("Germany") + vec("capital") ≈ vec("Berlin")

---

## Skip-Gram Model

Given a sequence of training words w₁, w₂, w₃ ... wₜ, the objective is to maximize the average log probability:

$$\frac{1}{T}\sum_{t=1}^{T}\;\sum_{\substack{-c \le j \le c \\ j \ne 0}} \log p(w_{t+j}\mid w_t)$$

Where **c** is the size of training context.

The probability is defined using the softmax function:

$$p(w_o \mid w_I) = \frac{\exp\!\left({v'_{w_o}}^{\top} v_{w_I}\right)} {\sum_{w=1}^{W} \exp\!\left({v'_w}^{\top} v_{w_I}\right)}$$

---

## Training Methods

### 1. Hierarchical Softmax

A computationally efficient approximation of the full softmax that uses a binary tree representation with W words as leaves:

$$p(w \mid w_I) = \prod_{j=1}^{L(w)-1} \sigma\!\left( \left[ n(w,j+1) = \mathrm{ch}(n(w,j)) \right] \cdot {v'_{n(w,j)}}^{\top} v_{w_I} \right)$$

### 2. Negative Sampling

Noise Contrastive Estimation (NCE) posits that a good model should differentiate data from noise via logistic regression. We can simplify NCE as **NEG**:

$$\log \sigma\!\left({v'_{w_o}}^{\top} v_{w_I}\right) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} \left[ \log \sigma\!\left(-{v'_{w_i}}^{\top} v_{w_I}\right) \right]$$

| Dataset Size | Recommended k |
|--------------|---------------|
| Small        | 5 – 20        |
| Large        | 2 – 5         |

### 3. Subsampling of Frequent Words

Common words like "the", "a" appear millions of times but add little value. Use subsampling where each word wᵢ is discarded with probability:

$$P(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}}$$

Where f(wᵢ) is the word frequency and t is a threshold (typically 10⁻⁵).

---

## Results

### Word Analogy Task

Example: "Germany" : "Berlin" :: "France" : ?

Solved by finding x where vec(x) is closest to:

> vec("Berlin") - vec("Germany") + vec("France")

---

## Learning Phrases

Many phrases have meanings not composable from individual words. To learn phrase vectors:

1. Find words that appear frequently together & infrequently in other contexts
2. Form many reasonable phrases without greatly increasing vocabulary size

**Scoring function** for phrase formation:

$$score(w_i, w_j) = \frac{count(w_i\ w_j) - \delta}{count(w_i) \times count(w_j)}$$

Where δ is a discounting coefficient preventing phrases with very infrequent words.

---

## Phrase Skip-Gram Results

Trained with dimensionality: **300**, context size: **5**

| Method   | No Subsampling | 10⁻⁵ Subsampling |
|----------|----------------|------------------|
| NEG-5    | 24%            | 27%              |
| NEG-15   | 27%            | 42%              |
| HS-Huffman | 19%          | **47%**          |

**Key Takeaways:**
- NEG with k=15 outperforms k=5 significantly
- Hierarchical Softmax is worst without subsampling, but **best** with subsampling
- Subsampling improves both speed **and** accuracy

---

## Additive Compositionality

Skip-gram vectors exhibit linear properties — words can be meaningfully combined via element-wise addition:

- vec("Russia") + vec("river") ≈ vec("Volga River")
- vec("Germany") + vec("capital") ≈ vec("Berlin")

### Why Does This Work?

Word vectors are trained to predict surrounding words, so they represent the **distribution of context** in which a word appears. The sum of two vectors relates to the product of their context distributions — words appearing frequently with *both* inputs get high probability. This acts like an **AND function**.

---

## Comparison to Other Models

| Model (training) | Redmond | Havel | graffiti | capitulate |
|-----------------|---------|-------|----------|------------|
| Collobert (50d, 2 months) | conyers, lubbock | plauen, osterreich | cheesecake, dioramas | abdicate, accede |
| Turian (200d, few weeks) | McCarthy, Alston | Jewell, Ovitz | gunfire, impunity | — |
| Mnih (100d, 7 days) | Podhurst, Harlang | Pontiff, Pinochet | anaesthetics | Mavericks |
| **Skip-Phrase** (1000d, 1 day) | **Redmond Wash., Microsoft** | **Vaclav Havel, Velvet Revolution** | **spray paint, grafitti** | **capitulation, capitulated** |

The Skip-gram model trained on **30B words** outperforms all others — trained in just **1 day** vs months for smaller models.

---

## Conclusion

### Key Contributions

- Distributed representations of **words and phrases** with linear structure enabling analogical reasoning
- Trained on orders of magnitude more data than previous models
- **Subsampling** → faster training + better rare word representations
- **Negative Sampling** → simple, accurate training for frequent words
- Hyperparameters matter: architecture, vector size, subsampling rate, window size

### Interesting Result

Word vectors can be meaningfully combined using **just vector addition**. Combined with phrase-as-single-token, this provides powerful text representation with minimal computational complexity.

> Code released as open-source.
