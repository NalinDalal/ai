## [Perceptron](https://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf)

> **Authors:** F. ROSENBLATT
> **Published in:** Psychological Review, Volume 65, Issue 6, November 1958
> **Significance:** groundbreaking work that fundamentally shaped the field of artificial intelligence. first artificial neural network capable of learning from data.

---

### 1. History Context
some scientist said that:
the central nervous system simply acts as an intricate switching network, where retention takes the form of new connections, or pathways, between centers of activity.

so theory has been developed
for a hypothetical nervous system, or
machine, called a perceptron

designed to illustrate
some of the fundamental properties of
intelligent systems in general, without
becoming too deeply enmeshed in the
special, and frequently unknown, conditions which hold for particular biological organisms

theory is based on following asumptions:
1. The physical connections of the
nervous system which are involved in
learning and recognition are not identical from one organism to another.
At birth, the construction of the most
important networks is largely random,
subject to a minimum number of
genetic constraints.
2. The original system of connected
cells is capable of a certain amount of
plasticity; after a period of neural
activity, the probability that a stimulus applied to one set of cells will
cause a response in some other set is
likely to change, due to some relatively long-lasting changes in the
neurons themselves.
3. Through exposure to a large
sample of stimuli, those which are
most "similar" (in some sense which
must be defined in terms of the
particular physical system) will tendto form pathways to the same sets of
responding cells. Those which are
markedly "dissimilar" will tend to
develop connections to different sets of
responding cells.
4. The application of positive and/
or negative reinforcement (or stimuli
which serve this function) may facilitate or hinder whatever formation of
connections is currently in progress.
5. Similarity, in such a system, is
represented at some level of the nervous system by a tendency of similar
stimuli to activate the same sets of
cells. Similarity is not a necessary
attribute of particular formal or geometrical classes of stimuli, but depends on the physical organization of
the perceiving system, an organization which evolves through interaction
with a given environment. The
structure of the system, as well as the
ecology of the stimulus-environment,
will affect, and will largely determine,
the classes of "things" into which the
perceptual world is divided.

# Organisation of Perceptron

The organization of a typical **photoperceptron** (responding to optical patterns) is shown in Figure 1 of the paper. The system has the following components:

### Sensory Units (S-points) — The Retina

- Stimuli impinge on a **retina** of sensory units (S-points)
- S-points respond on an **all-or-nothing** basis (binary activation)
- Some models use pulse amplitude/frequency proportional to stimulus intensity, but Rosenblatt assumes all-or-nothing

### Association Cells (A-units)

Impulses are transmitted from S-points to **association cells** (A-units) in two possible regions:

1. **Projection Area (A_I):** Each cell receives connections from S-points. The set of S-points transmitting to a particular A-unit = its **origin points** (excitatory or inhibitory). If the algebraic sum of impulse intensities ≥ threshold (θ), the A-unit fires (all-or-nothing). Origin points are **clustered/focalized** around a central point, falling off exponentially with retinal distance (supports contour detection).

2. **Association Area (A_II):** Connections from A_I to A_II are **random** — each A_II unit receives fibers from randomly scattered origin points in A_I. A_II units are otherwise identical to A_I units.

### Response Units (R-units)

- Responses R₁, R₂, ..., Rₙ are cells responding like A-units
- Each response has a large number of **origin points located at random** in the A_II set
- The A-units transmitting to a particular response = its **source-set**

### Feedback Connections

Up to A_II, all connections are **forward** (no feedback). Between A_II and R-units, connections go in **both directions**. Two rules for feedback:

- **(a)** Each response has **excitatory feedback** to cells in its own source-set (more anatomically plausible)
- **(b)** Each response has **inhibitory feedback** to the **complement** of its source-set (suppresses A-units that don't transmit to it — easier to analyze, used in most models)

### Value Dynamics (V)

Each A-unit has a **value V** (amplitude, frequency, latency, or probability of transmission). Higher value → more effective impulses. Value depends on metabolic condition of the cell. Activity tends to increase V; inactivity may decay it. Cells compete for metabolic materials.

Three systems investigated:

| System | Value Dynamics |
|--------|---------------|
| **α (alpha)** | Active cell gains a fixed increment per impulse; holds gain indefinitely |
| **β (beta)** | Each source-set has a constant rate of gain; increments apportioned among cells proportionally to activity |
| **γ (gamma)** | Active cells gain at the expense of inactive cells in the same source-set; total value of a source-set is **constant** |

### Two Phases of Response

1. **Predominant phase:** Some proportion of A-units respond to the stimulus, but R-units are still inactive (transient)
2. **Postdominant phase:** One response becomes dominant, inhibiting activity in the complement of its own source-set, preventing alternative responses

The initially dominant response is **random**, but after reinforcement (active units gain value), the same response will recur for the same stimulus → **learning**.

### Key Parameters

- **P_a** = expected proportion of A-units activated by a stimulus of given size
- **P_c** = conditional probability that an A-unit responding to stimulus S₁ also responds to S₂

For large retina (N_s → large), N_s ceases to be significant, and P_a, P_c approach values for an infinite retina.

### Six Basic Physical Parameters

All of learning, perceptual discrimination, and generalization can be predicted from:

| Parameter | Meaning |
|-----------|---------|
| **x** | Number of excitatory connections per A-unit |
| **y** | Number of inhibitory connections per A-unit |
| **θ** | Expected threshold of an A-unit |
| **ω** | Proportion of R-units to which an A-unit is connected |
| **N_A** | Number of A-units in the system |
| **N_R** | Number of R-units in the system |

---

# Mathematical Analysis of Learning in the Perceptron

### Two Response Systems

- **μ-system (mean-discriminating):** The response whose inputs have the **greatest mean value** responds first, gaining a slight advantage → quickly becomes dominant
- **Σ-system (sum-discriminating):** The response whose inputs have the **greatest net value** gains advantage

Systems responding to **mean values** generally have an advantage over **sum** systems, since means are less influenced by random variations in P_a.

### Learning Evaluation

Two experimental paradigms:

1. **Forced learning + same stimuli test:** Perceptron is forced to give desired responses during learning, then frozen. Same stimuli presented again → measure **P_r** (probability of correct choice)
2. **Forced learning + new stimuli from same classes:** After learning, new stimuli drawn from the same *classes* are presented → measure **P_g** (probability of correct generalization)

### General Approximation (Equation 4)

$$P = P(N_{a_r} > 0) \cdot \phi(Z)$$

where:

$$P(N_{a_r} > 0) = 1 - (1 - P_a)^{N_e}$$

$$\phi(Z) = \text{normal curve integral from } -\infty \text{ to } Z$$

$$Z = \frac{c_1 n_{s_r} + c_2}{\sqrt{c_3 n_{s_r}^2 + c_4 n_{s_r}}}$$

The four constants $c_1, c_2, c_3, c_4$ depend on the physical parameters of the perceptron and the stimulus environment.

### Ideal Environment (Random Stimuli)

For the **Σ-system** (α-system, fixed $n_{s_r}$):

$$c_1 = 0, \quad c_2 = (1 - P_a)N_e, \quad c_3 = 2P_a\omega, \quad c_4 \approx 0 \quad \text{...(5)}$$

For the **μ-system**:

$$c_1 = 0, \quad c_2 = (1 - P_a)N_e, \quad c_3 = 0, \quad c_4 = 2\omega \quad \text{...(6)}$$

**Key insight:** $c_3 = 0$ gives the μ-system a definite advantage over the Σ-system.

### Minimum of P_c

$$P_{c_{min}} = (1 - L)^x (1 - G)^y \quad \text{...(3)}$$

### Asymptotic Performance

After infinite experience with each class of stimuli:

$$P_{r\infty} = P_{g\infty} = [1 - (1 - P_a)^{N_e}] \times \phi\left(\frac{c_1}{\sqrt{c_3}}\right) \quad \text{...(9)}$$

This means *in the limit it makes no difference whether the perceptron has seen a particular test stimulus before or not* — if stimuli are drawn from a differentiated environment, performance will be equally good.

### Differentiated Environment

When stimuli are drawn from distinguishable classes (squares, circles, letters, etc.):

- $c_1 \neq 0$ → Z has a **nonrandom asymptote**
- If $P_{c11} > P_a > P_{c12}$: limiting performance $P_{g\infty}$ is better than chance → **generalization occurs**
- The generalization asymptote can be made **arbitrarily close to unity** by increasing number of A-units

#### Constants for Σ-system, α-perceptron in differentiated environment (Equation 10):

$$c_1 = P_a N_e (P_{c11} - P_{c12})$$
$$c_2 = P_a N_e (1 - P_{c11})$$
$$c_3 = \sum_{r=1,2} P_a(1-P_a)N_e \times [P_{c1r}^2 + \sigma_s^2(P_{c1r}) + \sigma_j^2(P_{c1r}) + (\omega N_R - 1)^2 \times (P_{c1x} + \sigma_s^2(P_{c1x}) + \sigma_j^2(P_{c1x})) + 2(\omega N_R - 1)(P_{c1r}P_{c1x})] + P_a^2 N_e^2 \times [\sigma_s^2(P_{c1r}) + (\omega N_R -1)^2 \sigma_s^2(P_{c1x}) + 2(\omega N_R - 1)\epsilon]$$

$$c_4 = \sum_{r=1,2} P_a N_e [P_{c1r} - P_{c1r}^2 - \sigma_s^2(P_{c1r}) - \sigma_j^2(P_{c1r}) + (\omega N_R - 1)(P_{c1x} - P_{c1x}^2 - \sigma_j^2(P_{c1x}))]$$

### Performance Comparison of Systems

From Figs. 7–10:
- **γ-system** ≥ α-system ≥ β-system in most conditions
- For the γ-system: $P_{r(\Sigma)} = P_{r(\mu)}$ (no difference between sum and mean discrimination)
- α and β systems perform **better with μ-system** (mean-discriminating) than Σ-system

---

# Bivalent Systems

In all systems above, value increments were always **positive** (monovalent). In a **bivalent system**, two types of reinforcement exist:

- **Positive reinforcement:** +ΔV added to active A-units in source-sets of "on" responses; −ΔV to "off" response source-sets
- **Negative reinforcement:** −ΔV to active units in "on" source-sets; +ΔV to "off" source-sets

Bivalent systems are efficient at reducing **bias effects** (preference for wrong response due to size/frequency of associated stimuli).

A bivalent γ-system with disjunct source-sets has the **same coefficients** as the monovalent α-system for the μ-case (Equation 11).

---

# Improved Perceptrons and Spontaneous Organization

- A **momentary stimulus perceptron** has no temporal pattern recognition
- If A-unit values are allowed to **decay proportionally** to activity rate → activity at time $t$ depends on activity at $t-1$ → **temporal pattern recognition** becomes possible
- With suitable origin point organization (constrained spatial distribution), A-units become **sensitive to contour location** → improved performance
- With proper reinforcement: **spontaneous concept formation** — exposing the system to "dissimilar" stimulus classes with automatic reinforcement leads to stable binary responses ("1" for one class, "0" for the other) without explicit forcing

### Capabilities

- Pattern recognition (spatial and temporal)
- Associative learning & selective recall
- Selective attention via cognitive sets
- Trial-and-error learning (bivalent systems)
- Learning ordered sequences of responses (with sensory feedback)

### Limitations

- Recognition of **relationships** in space/time → limit to cognitive abstraction
- **Statistical separability alone** is insufficient for higher-order abstraction
- A more advanced system than the perceptron is needed for relational/symbolic reasoning

---

# Conclusions and Evaluation

1. In an **ideal environment** of random stimuli, a randomly connected system can learn to associate specific responses to specific stimuli (better-than-chance), even with mutual resemblance
2. In an ideal environment, probability of correct response **diminishes toward 0.5** (random) as number of stimuli increases
3. In an ideal environment, **no basis for generalization** exists
4. In a **differentiated environment** (each response → distinct class of correlated stimuli), retention approaches a **better-than-chance asymptote** as stimuli increase; asymptote can approach unity by increasing A-units
5. In a differentiated environment, probability that an **unseen stimulus** will be correctly classified (**generalization**) approaches the same asymptote as retention — if $P_{c12} < P_a < P_{c11}$
6. Performance can be improved by: **contour-sensitive projection area**, **binary response system** (each "bit" = independent feature)
7. **Trial-and-error learning** is possible in bivalent reinforcement systems
8. **Temporal organization** of stimulus patterns and responses can be learned via extension of statistical separability principles
9. Memory is **distributed** — removal of a portion degrades *all* associations slightly rather than destroying any single one
10. Simple cognitive sets, selective recall, and **spontaneous class recognition** are possible; but **relational abstraction** is a limit

### Merits of the Theory

| Criterion | Description |
|-----------|-------------|
| **Parsimony** | Only one hypothetical variable (V, cell value) needed beyond basic physics |
| **Verifiability** | All 6 parameters are independently measurable physical variables |
| **Explanatory power** | Derived from basic physics → generalizable to any learning system with known physical parameters |

---

# Simulation: Perceptron on AND/OR Gates

> See `perceptron_and_or.py` for the implementation.

--------