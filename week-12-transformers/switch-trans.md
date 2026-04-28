# [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/pdf/2101.03961)

Basically Mixture of Expert architecture.

Switch Transformer: improves over MoE architecture

Improved training over limited resources.

Reduce model size by 99%, while preserving 30% quality gains

Improved training techniques: selective precision training allowed with lower float16 precision; allows to scale to large no of mix of expert; increased expert regularization that improves sparse model fine-tuning and multi-task training.

Improvement over multiple languages

model scaling increased by upto trillion parameter.

# Switch Transformers

Benefits of scaling was studied by some researchers which uncovered powerlaw scaling with `model size`, `data set size` and `computational budget`.
They suggested that raining large models on relatively small amounts of data is the computationally optimal approach.

let's check a 4th axis:  increase the *parameter count* while keeping the floating point operations (FLOPs) per example constant.

Our hypothesis is
that the parameter count, independent of total computation performed, is a separately
important axis on which to scale. We achieve this by designing a sparsely activated model
that efficiently uses hardware designed for dense matrix multiplications such as GPUs and
TPUs. Our work here focuses on TPU architectures, but these class of models may be
similarly trained on GPU clusters. In our distributed training setup, our sparsely activated
layers split unique weights on different devices. Therefore, the weights of the model increase
with the number of devices, all while maintaining a manageable memory and computational
footprint on each device.

![encoder-block](./switch-transformer-encoder.png)


##  Simplifying Sparse Routing
We say that we take a input token representation $x$ and then feed it to best determined top-k expert from a set {Ei(x)} i=1 to N of N experts.
h(x)=W\sub{r} * x normalised via softmax over N experts.

gate value of ith expert:
p\sub{i}(x)=\frac{e^{h(i)}}{\sum{j}{N}{e^{h(x)\sub{j}}}}

The top-k gate values are selected for routing the token x. If T is the set of selected top-k
indices then the output computation of the layer is the linearly weighted combination of
each expert’s computation on the token by the gate value,
y = \sigma{i∈T}{pi(x)Ei(x)}.

study the top-k decision and found that higher k-values in lower layers in the model were
important for models with many routing layers.

a simplified strategy where we route to only a single expert. We show this simplification
preserves model quality, reduces routing computation and performs better. This k = 1
routing strategy is later referred to as a Switch layer.

benefits are 3 fold:
(1) The router computation is reduced
as we are only routing a token to a single expert. 
(2) The batch size (expert capacity) of each expert can be at least halved since each token is only being routed to a single expert.
(3) The routing implementation is simplified and communication costs are reduced.

batch size of expert: (total tokens / num experts) × capacity factor.

## Efficient Sparse Routing
use [MTF](https://github.com/tensorflow/mesh) to efficient distributed data and model parallel architectures.

**Distributed Switch Implementation**
tensor are static but the computation is dynamic. so question arises: how to set expert capacity.
expert capacity—the number of tokens each expert computes
set by evenly dividing the number of tokens in the batch across the number of experts, and
then further expanding by a *capacity factor*

$expert capacity= \frac{tokens per batch}{number of experts} * capacity factor$.



----

7-35
