# [**Distributed Representations of Words and Phrases and their Compositionality**](https://arxiv.org/pdf/1310.4546)

# Introduction
Distributed representations of words in a vector space help learning algorithms to achieve better
performance in natural language processing tasks by grouping similar words. 
Skip-gram Model says to represent words in vectors from unstructured texts.
This does not involves dense matrix multiplications.
They are interesting via Neural Network cause learned vectors explicitly encode many linguistic regularities and patterns.
ex:  vec("Madrid") - vec("Spain") + vec("France") is closer to vec("Paris") 
Word representations are limited by their inability to represent idiomatic phrases that are not compositions of the individual words. For example, "Boston Globe" is a newspaper, and so it is not a
natural combination of the meanings of "Boston" and "Globe".
hence expensive to use Skip Gram Model
We found that simple vector addition can often produce meaningful results. 
For example, vec("Russia") + vec("river") is close to vec("Volga River"), and vec("Germany") + vec("capital") is close to vec("Berlin").

# Skip-Gram Model
given a sequence of training words w1,w2,w3...wT, the objective of Skip-Gram model is to maximise average log probability
$$
\frac{1}{T}\sum_{t=1}^{T}\;\sum_{\substack{-c \le j \le c \\ j \ne 0}} \log p(w_{t+j}\mid w_t)
$$
c - size of training context
formulation defines via softmax function
$$
p(w_o \mid w_I)=
\frac{\exp\!\left({v'_{w_o}}^{\top} v_{w_I}\right)}
{\sum_{w=1}^{W} \exp\!\left({v'_w}^{\top} v_{w_I}\right)}
$$

## Hierarchical softmax
it is A computationally efficient approximation of the full softmax
uses a binary tree representation of the output layer with the W words as its leaves and, 
for each node, explicitly represents the relative probabilities of its child nodes. 
$$
p(w \mid w_I)=
\prod_{j=1}^{L(w)-1}
\sigma\!\left(
\left[ n(w,j+1) = \mathrm{ch}(n(w,j)) \right]
\cdot {v'_{n(w,j)}}^{\top} v_{w_I}
\right)
$$

## Negative Sampling
Noise Contrastive Estimation (NCE) posits that a good model should be able to differentiate data from noise by means of logistic
regression.
we are free to
simplify NCE as long as the vector representations retain their quality
define Negative sampling(NEG) by the objective
$$
\log \sigma\!\left({v'_{w_o}}^{\top} v_{w_I}\right)
+
\sum_{i=1}^{k}
\mathbb{E}_{w_i \sim P_n(w)}
\left[
\log \sigma\!\left(-{v'_{w_i}}^{\top} v_{w_I}\right)
\right]
$$

k: 5 to 20 for small dataset

k: 2 to 5 for large dataset 

## Sub Sampling of Frequent Words
common words like 'a', 'the' etc they appear more than million times => adds no value
the vector representations of frequent words do not change significantly after training on several million examples
to counter this imbalance
used a simple subsampling approach: each word $w_i$ in the training set is discarded with probability computed by the formula
$$
P(w_i)=1-\sqrt{\frac{t}{f(w_i)}}
$$
f(wi) is the frequency of word wi and t is a chosen threshold, typically around {10}^{−5}.

# Results
consider an analogy: "Germany" : "Berlin" :: "France" : ?
<br>
solved by finding a vector x such that vec(x) is closest to vec("Berlin") - vec("Germany")+ vec("France")

# Learning Phrases
many phrases have a meaning that is not a simple composition of the meanings of its individual words.
vector representation for phrases:
- find words that appear frequently together & infrequently in other contexts
- form many reasonable phrases without greatly increasing the size of the vocabulary

use a simple data-driven approach, where phrases are formed based on the unigram and bigram counts, using
$$
score(w_i,w_j)=\frac{count(w_i\ w_j)-\delta}{count(w_i)\times count(w_j)}
$$
$\delta$ is a discounting coefficient and prevents too many phrases consisting of very infrequent words to be formed

## Phrase Skip-Gram Results

trained several Skip-gram models with different hyperparameters on the same news data, using:
- vector dimensionality: 300
- context size: 5

compared **Negative Sampling** vs **Hierarchical Softmax**, both with and without subsampling of frequent tokens.

| Method | Dimensionality | No subsampling [%] | 10⁻⁵ subsampling [%] |
|---|---|---|---|
| NEG-5 | 300 | 24 | 27 |
| NEG-15 | 300 | 27 | 42 |
| HS-Huffman | 300 | 19 | **47** |

key takeaways:
- NEG with k=15 does considerably better than k=5
- Hierarchical Softmax performs worst without subsampling, but becomes the **best** method when subsampling is applied
- subsampling can speed up training **and** improve accuracy at the same time

# Additive Compositionality

Skip-gram vectors show another cool linear property — words can be meaningfully combined using simple element-wise addition of their vectors.

for example:
- vec("Russia") + vec("river") ≈ vec("Volga River")
- vec("Germany") + vec("capital") ≈ vec("Berlin")

**why does this work?**
word vectors are in a linear relationship with the inputs to the softmax. since vectors are trained to predict surrounding words, they represent the *distribution of context* in which a word appears. the sum of two vectors is related to the product of their context distributions — words that appear frequently with *both* input words will get high probability. this acts like an AND function.

so if "Volga River" appears often in sentences with both "Russian" and "river", then vec("Russian") + vec("river") ends up close to vec("Volga River").

# Comparison to Published Word Representations

other well-known models: Collobert & Weston, Turian et al., Mnih & Hinton — all published their word vectors.

Skip-gram model was already shown to outperform these on the word analogy task by a huge margin.

to further compare quality, nearest neighbours of infrequent words were inspected across models:

| Model (training time) | Redmond | Havel | ninjutsu | graffiti | capitulate |
|---|---|---|---|---|---|
| Collobert (50d, 2 months) | conyers, lubbock, keene | plauen, dzerzhinsky, osterreich | reiki, kohona, karate | cheesecake, gossip, dioramas | abdicate, accede, rearm |
| Turian (200d, few weeks) | McCarthy, Alston, Cousins | Jewell, Arzu, Ovitz | - | gunfire, emotion, impunity | - |
| Mnih (100d, 7 days) | Podhurst, Harlang, Agarwal | Pontiff, Pinochet, Rodionov | - | anaesthetics, monkeys, Jews | Mavericks, planning, hesitated |
| **Skip-Phrase** (1000d, 1 day) | **Redmond Wash., Redmond Washington, Microsoft** | **Vaclav Havel, president Vaclav Havel, Velvet Revolution** | **ninja, martial arts, swordsmanship** | **spray paint, grafitti, taggers** | **capitulation, capitulated, capitulating** |

the big Skip-gram model trained on 30B words clearly outperforms all others — and it only took **1 day** to train vs months for smaller models. this is because it was trained on ~30B words, which is 2–3 orders of magnitude more data than prior work, yet the training time is just a fraction of what older architectures needed.

# Conclusion

key contributions of this paper:

- showed how to train distributed representations of words **and phrases** with Skip-gram, with linear structure that enables precise analogical reasoning
- successfully trained on several orders of magnitude more data than previous models, thanks to the computationally efficient architecture
- **subsampling** frequent words → faster training + better representations of rare words
- **Negative Sampling** → simple training method that learns accurate representations especially for frequent words
- hyperparameter choices matter a lot: model architecture, vector size, subsampling rate, training window size are the most crucial decisions

interesting result: word vectors can be meaningfully combined using **just vector addition**. combining this with the phrase-as-single-token approach gives a powerful yet simple way to represent longer text with minimal computational complexity. this complements existing approaches like recursive matrix-vector operations.

code was released as an open-source project.
