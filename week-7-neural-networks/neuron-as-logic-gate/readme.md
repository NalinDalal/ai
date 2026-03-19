## [McCulloch & Pitts (1943) — "A Logical Calculus of the Ideas Immanent in Nervous Activity"](https://www.cs.cmu.edu/~epxing/Class/10715/reading/McCulloch.and.Pitts.pdf)

> **Authors:** Warren McCulloch (neurophysiologist) & Walter Pitts (logician)
> **Published in:** The Bulletin of Mathematical Biophysics, Vol 5(4), pp 115-133
> **Significance:** First mathematical model of the nervous system as a network of simple logical elements (artificial neurons). Foundational result in automata theory, cognitive science, computational neuroscience, and AI. Cited by John von Neumann as a significant result.

---

### 1. Historical Context

- **Turing (1937)** formalized computation via Turing Machines — an infinite tape, a tape head, and a finite instruction table operating on binary cells (0 or 1).
- Alternative models of computation already existed: lambda calculus, cellular automata.
- McCulloch & Pitts were inspired to ask: can biological neurons also be described as computational devices?
- **Walter Pitts** (ran away from home at 15, arrived at University of Chicago) studied under **Rudolf Carnap**, attended **Rashevsky's** seminars on mathematical biology.
- **Warren McCulloch** was interested in circular causality from studies with causalgia, epileptic activity, and Lorente de Nò's recurrent neural networks research.
- Pitts provided the mathematical and logical rigor (symbolic logic, modular arithmetic) to McCulloch's vague ideas on "psychons."
- The paper used **Carnap's "Language II"** from *The Logical Syntax of Language* with notations from **Whitehead & Russell's** *Principia Mathematica*.

---

### 2. Key Biological Insight

Neurons have three main components:
1. **Cell body** — contains the nucleus and metabolic machinery
2. **Axon** — transmits information via synaptic terminals
3. **Dendrites** — receive inputs from other neurons via synapses

**How neurons work:**
- Neurons communicate by passing **electro-chemical signals** from axon terminals (pre-synaptic) to dendrites (post-synaptic)
- Each neuron connects to hundreds/thousands of other neurons
- For a neuron to **fire**, a certain **voltage threshold** must be passed
- The combined excitatory and inhibitory input determines whether the neuron fires
- This **all-or-none** firing behavior is what McCulloch & Pitts modeled computationally

---

### 3. The McCulloch-Pitts (M-P) Neuron Model — 5 Core Rules

1. **Neuron activation is binary** — a neuron either fires (1) or does not fire (0)
2. **Weighted sum threshold** — for a neuron to fire, the weighted sum of inputs must be ≥ a predefined threshold $T$
3. **Inhibitory veto** — if one or more inputs are inhibitory, the neuron will not fire
4. **Fixed time step** — it takes exactly one time step for a signal to pass through a link
5. **No learning** — neither the structure nor the weights change over time

---

### 4. Mathematical Formulation (Linear Threshold Gate)

$$Sum = \sum_{i=1}^{N} I_i \cdot W_i$$

$$y(Sum) = \begin{cases} 1 & \text{if } Sum \geq T \\ 0 & \text{otherwise} \end{cases}$$

Where:
- $I_1, I_2, \ldots, I_N$ are **binary input values** $\in \{0, 1\}$
- $W_1, W_2, \ldots, W_N$ are **weights** $\in \{-1, 1\}$
- $Sum$ is the **weighted sum** of inputs
- $T$ is the **predefined threshold** for neuron activation
- $y$ is the **output** (fire or not-fire)

**Excitatory vs Inhibitory:**
- **Excitatory input:** $I_i \cdot W_i = 1 \cdot 1 = 1$ (positive contribution)
- **Inhibitory input:** $I_i \cdot W_i = 1 \cdot (-1) = -1$ (negative contribution)

The output function is a **step function** (Heaviside):

$$N_i(t+1) = H\left(\sum_{j=1}^{n} w_{ij}(t) \cdot N_j(t) - \theta_i(t)\right)$$

> **Important:** In the M-P model, weights only determine whether an input is excitatory or inhibitory ($+1$ or $-1$). This is different from modern neural networks where weights scale input values continuously.

---

### 5. Python Implementation

```python
import numpy as np

# Step 1: Generate inputs and weights
np.random.seed(seed=0)
I = np.random.choice([0, 1], 3)    # binary inputs
W = np.random.choice([-1, 1], 3)   # weights ∈ {-1, 1}

# Step 2: Compute dot product (weighted sum)
dot = I @ W

# Step 3: Define threshold activation function
def linear_threshold_gate(dot: int, T: float) -> int:
    """Returns 1 if weighted sum >= threshold, else 0"""
    if dot >= T:
        return 1
    else:
        return 0

# Step 4: Compute output
T = 1
activation = linear_threshold_gate(dot, T)  # → 1 if dot >= T
```

---

### 6. Boolean Logic Gates with M-P Neurons

#### 6a. AND Gate
| $I_1$ | $I_2$ | Output |
|-------|-------|--------|
| 0     | 0     | 0      |
| 0     | 1     | 0      |
| 1     | 0     | 0      |
| 1     | 1     | **1**  |

**Config:** Weights = $[1, 1]$, Threshold $T = 2$
- Fires only when **all** inputs are excitatory

```python
input_table = np.array([[0,0], [0,1], [1,0], [1,1]])
weights = np.array([1, 1])
dot_products = input_table @ weights   # → [0, 1, 1, 2]
T = 2
# outputs: 0, 0, 0, 1 ✓
```

#### 6b. OR Gate
| $I_1$ | $I_2$ | Output |
|-------|-------|--------|
| 0     | 0     | 0      |
| 0     | 1     | **1**  |
| 1     | 0     | **1**  |
| 1     | 1     | **1**  |

**Config:** Weights = $[1, 1]$, Threshold $T = 1$
- Fires when **at least one** input is excitatory

```python
T = 1
# same dot_products [0, 1, 1, 2]
# outputs: 0, 1, 1, 1 ✓
```

#### 6c. NOR Gate
| $I_1$ | $I_2$ | Output |
|-------|-------|--------|
| 0     | 0     | **1**  |
| 0     | 1     | 0      |
| 1     | 0     | 0      |
| 1     | 1     | 0      |

**Config:** Weights = $[-1, -1]$, Threshold $T = 0$
- Fires only when **all** inputs are inhibitory (both off)

```python
weights = np.array([-1, -1])
dot_products = input_table @ weights   # → [0, -1, -1, -2]
T = 0
# outputs: 1, 0, 0, 0 ✓
```

#### Summary — Gate Configurations
| Gate | Weights | Threshold |
|------|---------|-----------|
| AND  | all positive (+1) | 2 (# of inputs) |
| OR   | all positive (+1) | 1 |
| NOR  | all negative (-1) | 0 |
| NOT  | negative (-1) | 0 (single input) |

---

### 7. Theorems from the Paper

- **Theorems 1 & 2:** Neural nets **without loops** (acyclic) are equivalent to **Temporal Propositional Expressions (TPE)** — i.e., propositional formulas with one free variable $t$.
  - Example: $N_1(t) \lor N_2(t) \land \lnot N_3(t)$

- **Theorems 4 & 5:** Three forms of inhibition are equivalent:
  - **Relative inhibition** — weighted sum must exceed threshold
  - **Absolute inhibition** — any negative synapse firing prevents neuron firing
  - **Extinction** — inhibitory firing temporarily raises threshold

- **Theorem 6:** Three forms of excitation are equivalent:
  - **Spatial summation** — multiple synapses fire close together, effects sum
  - **Temporal summation** — signal accumulates over time window $T$
  - **Facilitation** — temporary lowering of threshold

- **Theorem 7:** Networks with fixed synapses and networks with **Hebbian learning** (latent synapses activate when both neurons fire simultaneously) are equivalent in computational power.

- **Theorems 8-10:** Neural nets **with loops** can encode all **first-order logic with equality** — and conversely, every looped neural network is equivalent to a sentence in first-order logic with equality.

- **Key result:** A neural network, if furnished with a tape, scanners, and write-heads, is **equivalent to a Turing machine** (and vice versa). Thus M-P networks are equivalent to **Turing computability** and **Church's lambda-definability**.

---

### 8. Limitations of the M-P Neuron

1. **Only binary I/O** — real-world features are often continuous; a probability-valued output would be more useful
2. **No learning** — you must figure out the solution (weights, threshold) yourself beforehand; zero autonomy
3. **Manual parameter tuning** — weights and thresholds must be hand-set
4. **Oversimplified biology** — real neurons exhibit far more complex behavior (graded potentials, neurotransmitter diversity, dendritic computation, etc.)

---

### 9. Legacy & Subsequent Work

- **John von Neumann** cited this as a significant result in automata theory
- **Kleene (1951)** proved that **regular languages** = exactly what M-P neural nets can generate → term "regular" comes from "regularly occurring events"
- **Marvin Minsky** was directly influenced → built **SNARC** (1951, early neural network hardware) → PhD on neural networks (1954)
- **McCulloch** chaired the 10 **Macy Conferences** (1946-1953) → birth of **cybernetics** and **cognitive science**
- **Pitts & McCulloch (1947)** — "How We Know Universals" → generalized to spatial object recognition
- **Norbert Wiener** found this evidence for a general method of animal object recognition via scanning/transformations
- Led directly to **Rosenblatt's Perceptron (1958)** — which added **learning** to the neuron model

---

### 10. Key Takeaway

The McCulloch-Pitts neuron was the **first formal model showing that neural networks can perform computation**. It proved that networks of simple binary threshold units can implement any logical function and are Turing-complete. However, it lacks the two things that make modern neural networks powerful: (1) **continuous-valued activations** and (2) a **learning algorithm**. These limitations were addressed by later work — Rosenblatt's Perceptron (1958) added learning, and the backpropagation algorithm (Rumelhart et al., 1986) enabled training of deep multi-layer networks.

---

### 11. [Implementation](./init.py)