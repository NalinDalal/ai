"""
McCulloch & Pitts (1943) — "A Logical Calculus of the Ideas Immanent in Nervous Activity"
=========================================================================================

Implementation of the McCulloch-Pitts (M-P) artificial neuron model or logical gate network (AND, OR, NOT using M-P neurons).

The M-P neuron is the first mathematical model of a biological neuron, proposed by
Warren McCulloch (neurophysiologist) and Walter Pitts (logician) in 1943. It models
neurons as simple binary threshold units that can implement logical functions.

Core principles:
    1. Neuron activation is binary — fire (1) or not-fire (0)
    2. A neuron fires when the weighted sum of inputs >= threshold T
    3. If any inhibitory input is active, the neuron does NOT fire (absolute inhibition)
    4. Signals propagate in fixed, discrete time steps
    5. Weights and structure do not change (no learning)

Mathematical formulation (Linear Threshold Gate):
    Sum = Σ (I_i * W_i)  for i = 1..N
    y   = 1 if Sum >= T, else 0

Where:
    I_i ∈ {0, 1}   — binary inputs
    W_i ∈ {-1, +1}  — excitatory (+1) or inhibitory (-1) weights
    T               — firing threshold

Reference:
    McCulloch, W.S. & Pitts, W. (1943). A logical calculus of the ideas immanent
    in nervous activity. Bulletin of Mathematical Biophysics, 5(4), 115-133.
"""

import numpy as np
from typing import List, Tuple


# =============================================================================
# 1. THE M-P NEURON — Core Building Block
# =============================================================================

class MPNeuron:
    """
    A single McCulloch-Pitts neuron.

    The neuron computes a weighted sum of binary inputs and fires (outputs 1)
    if the sum meets or exceeds a threshold T. If any inhibitory input is active,
    the neuron is suppressed and will not fire (absolute inhibition rule from the
    original paper — Theorem 4).

    Attributes:
        weights (np.ndarray): Weight vector, each element ∈ {-1, +1}.
                              +1 = excitatory synapse, -1 = inhibitory synapse.
        threshold (float):    Firing threshold T. Neuron fires when weighted sum >= T.
        name (str):           Optional label for the neuron.
    """

    def __init__(self, weights: List[int], threshold: float, name: str = "MPNeuron"):
        """
        Initialize an M-P neuron.

        Args:
            weights:   List of integer weights, each should be +1 (excitatory)
                       or -1 (inhibitory).
            threshold: The firing threshold T.
            name:      Human-readable name for display/debugging.

        Raises:
            ValueError: If any weight is not in {-1, +1}.
        """
        self.weights = np.array(weights)
        self.threshold = threshold
        self.name = name

        # Validate that weights follow the M-P constraint: only {-1, +1}
        if not all(w in (-1, 1) for w in self.weights):
            raise ValueError(
                f"M-P neuron weights must be ∈ {{-1, +1}}, got {weights}. "
                "In the original 1943 model, weights only encode excitatory/inhibitory — "
                "they do not scale input magnitudes."
            )

    def activate(self, inputs: np.ndarray) -> int:
        """
        Compute the neuron's output for a given input vector.

        This implements the linear threshold gate:
            1. Compute weighted sum: Sum = I · W
            2. Apply Heaviside step function: y = 1 if Sum >= T, else 0

        The original paper also described "absolute inhibition" (Theorem 4):
        if ANY inhibitory input is active (I_i=1 and W_i=-1), the neuron does
        not fire regardless of excitatory inputs. We implement this as a flag.

        Args:
            inputs: Binary input vector, each element ∈ {0, 1}.

        Returns:
            1 if neuron fires, 0 otherwise.
        """
        inputs = np.array(inputs)

        # --- Absolute inhibition check (from Theorems 4-5 of the paper) ---
        # If any inhibitory synapse has an active input, neuron is silenced.
        inhibitory_mask = self.weights < 0
        if np.any(inputs[inhibitory_mask] == 1):
            return 0

        # --- Weighted sum (spatial summation) ---
        weighted_sum = np.dot(inputs, self.weights)

        # --- Heaviside step function ---
        # N_i(t+1) = H(Σ w_ij * N_j(t) - θ_i)
        return 1 if weighted_sum >= self.threshold else 0

    def truth_table(self) -> List[Tuple[tuple, int]]:
        """
        Generate the complete truth table for this neuron.

        Enumerates all 2^N possible binary input combinations and computes
        the output for each. Useful for verifying logical gate behavior.

        Returns:
            List of (input_tuple, output) pairs.
        """
        n = len(self.weights)
        table = []
        for i in range(2**n):
            # Generate binary input vector from integer i
            inputs = np.array([(i >> bit) & 1 for bit in range(n - 1, -1, -1)])
            output = self.activate(inputs)
            table.append((tuple(inputs), output))
        return table

    def print_truth_table(self):
        """Pretty-print the truth table with column headers."""
        n = len(self.weights)
        table = self.truth_table()

        # Header
        input_headers = [f"I{j+1}" for j in range(n)]
        header = " | ".join(input_headers + ["Out"])
        separator = "-+-".join(["--"] * (n + 1))
        print(f"\n{self.name} (weights={list(self.weights)}, T={self.threshold})")
        print(header)
        print(separator)

        # Rows
        for inputs, output in table:
            row = " | ".join([f" {v}" for v in inputs] + [f" {output}"])
            print(row)
        print()

    def __repr__(self):
        return f"MPNeuron(name='{self.name}', weights={list(self.weights)}, T={self.threshold})"


# =============================================================================
# 2. STANDARD LOGIC GATES — Built from M-P Neurons
# =============================================================================

def AND_gate() -> MPNeuron:
    """
    AND gate: fires only when ALL inputs are active.

    Configuration:
        - Weights: [+1, +1] (both excitatory)
        - Threshold: 2 (sum must equal number of inputs)

    Truth table:
        (0,0) → 0    (0,1) → 0    (1,0) → 0    (1,1) → 1
    """
    return MPNeuron(weights=[1, 1], threshold=2, name="AND")


def OR_gate() -> MPNeuron:
    """
    OR gate: fires when AT LEAST ONE input is active.

    Configuration:
        - Weights: [+1, +1] (both excitatory)
        - Threshold: 1 (only one excitatory input needed)

    Truth table:
        (0,0) → 0    (0,1) → 1    (1,0) → 1    (1,1) → 1
    """
    return MPNeuron(weights=[1, 1], threshold=1, name="OR")


def NOT_gate() -> MPNeuron:
    """
    NOT gate (inverter): fires when the single input is OFF.

    Configuration:
        - Weights: [-1] (inhibitory)
        - Threshold: 0 (fires by default unless inhibited)

    Truth table:
        (0,) → 1    (1,) → 0
    """
    return MPNeuron(weights=[-1], threshold=0, name="NOT")


def NOR_gate() -> MPNeuron:
    """
    NOR gate: fires only when ALL inputs are OFF.
    Equivalent to NOT(OR(I1, I2)).

    Configuration:
        - Weights: [-1, -1] (both inhibitory)
        - Threshold: 0 (fires when no input is active)

    Truth table:
        (0,0) → 1    (0,1) → 0    (1,0) → 0    (1,1) → 0
    """
    return MPNeuron(weights=[-1, -1], threshold=0, name="NOR")


class NAND_gate:
    """
    NAND gate: fires unless ALL inputs are active.
    Equivalent to NOT(AND(I1, I2)).

    A single M-P neuron with absolute inhibition CANNOT implement NAND
    because inhibitory inputs trigger a veto even when only one is active.
    So we compose two neurons: AND → NOT.

    Truth table:
        (0,0) → 1    (0,1) → 1    (1,0) → 1    (1,1) → 0
    """

    def __init__(self):
        self.name = "NAND"
        self._and = AND_gate()
        self._not = NOT_gate()

    def activate(self, inputs):
        and_out = self._and.activate(inputs)
        return self._not.activate([and_out])

    def print_truth_table(self):
        n = 2
        print(f"\n{self.name} (composed: AND → NOT)")
        print("I1 | I2 | Out")
        print("---+----+---")
        for a in [0, 1]:
            for b in [0, 1]:
                out = self.activate(np.array([a, b]))
                print(f" {a} |  {b} |  {out}")
        print()


# =============================================================================
# 3. COMPOSITE LOGIC — Multi-Neuron Networks (XOR, XNOR)
# =============================================================================

class MPNetwork:
    """
    A network of interconnected McCulloch-Pitts neurons.

    The original paper (Theorems 1-2) showed that acyclic networks of M-P neurons
    can compute any Temporal Propositional Expression. With loops (Theorems 8-10),
    they become equivalent to first-order logic and Turing machines.

    This class implements feed-forward (acyclic) networks for combining gates.
    """

    def __init__(self, name: str = "MPNetwork"):
        self.name = name
        self.layers = []  # List of (neurons, connections) per layer

    def compute(self, inputs: np.ndarray) -> int:
        """Override in subclasses to define network topology."""
        raise NotImplementedError


class XOR_network(MPNetwork):
    """
    XOR gate: fires when exactly one input is active.

    XOR CANNOT be implemented by a single M-P neuron — this is a key limitation.
    It requires a network of neurons (multi-layer architecture).

    Implementation using the identity: XOR(A,B) = AND(OR(A,B), NAND(A,B))
    Alternatively: XOR(A,B) = OR(AND(A, NOT(B)), AND(NOT(A), B))

    We use the second form with 5 neurons:
        Layer 1: NOT_A, NOT_B (two NOT gates)
        Layer 2: AND(A, NOT_B), AND(NOT_A, B) (two AND-like gates)
        Layer 3: OR of Layer 2 outputs

    Truth table:
        (0,0) → 0    (0,1) → 1    (1,0) → 1    (1,1) → 0
    """

    def __init__(self):
        super().__init__(name="XOR")
        # Layer 1: Inverters
        self.not_a = NOT_gate()
        self.not_b = NOT_gate()
        # Layer 2: AND gates (using excitatory weights, T=2)
        self.and_1 = AND_gate()  # AND(A, NOT_B)
        self.and_2 = AND_gate()  # AND(NOT_A, B)
        # Layer 3: OR gate
        self.or_out = OR_gate()  # OR(and_1, and_2)

    def compute(self, inputs: np.ndarray) -> int:
        """
        Compute XOR through a 3-layer feed-forward network.

        Signal flow (each arrow = 1 time step, per M-P rule #4):
            t=0: inputs A, B arrive
            t=1: NOT gates produce ~A, ~B
            t=2: AND gates produce AND(A, ~B) and AND(~A, B)
            t=3: OR gate produces final output

        Args:
            inputs: Binary array [A, B] where A, B ∈ {0, 1}.

        Returns:
            1 if exactly one input is active, else 0.
        """
        A, B = int(inputs[0]), int(inputs[1])

        # Layer 1: inversions (time step t+1)
        not_A = self.not_a.activate([A])
        not_B = self.not_b.activate([B])

        # Layer 2: conjunctions (time step t+2)
        and_a_notb = self.and_1.activate([A, not_B])
        and_nota_b = self.and_2.activate([not_A, B])

        # Layer 3: disjunction (time step t+3)
        result = self.or_out.activate([and_a_notb, and_nota_b])
        return result

    def truth_table(self) -> List[Tuple[tuple, int]]:
        """Generate complete truth table for XOR network."""
        table = []
        for a in [0, 1]:
            for b in [0, 1]:
                out = self.compute(np.array([a, b]))
                table.append(((a, b), out))
        return table

    def print_truth_table(self):
        """Pretty-print the XOR truth table."""
        print(f"\n{self.name} Network (multi-layer M-P neurons)")
        print("I1 | I2 | Out")
        print("---+----+----")
        for inputs, output in self.truth_table():
            print(f" {inputs[0]} |  {inputs[1]} |  {output}")
        print()


# =============================================================================
# 4. DEMONSTRATION — Reproducing the Paper's Key Results
# =============================================================================

def demo_single_neuron():
    """
    Demonstrate basic M-P neuron operation.

    From the paper's mathematical formulation:
        Sum = Σ I_i * W_i
        y = 1 if Sum >= T, else 0
    """
    print("=" * 60)
    print("DEMO 1: Single M-P Neuron — Basic Operation")
    print("=" * 60)

    # Create a neuron with 3 excitatory inputs, threshold = 2
    neuron = MPNeuron(weights=[1, 1, 1], threshold=2, name="3-input neuron")
    print(f"\n{neuron}")
    print("This neuron fires when at least 2 of 3 inputs are active.\n")

    test_cases = [
        ([0, 0, 0], "All off"),
        ([1, 0, 0], "One on"),
        ([1, 1, 0], "Two on"),
        ([1, 1, 1], "All on"),
    ]

    for inputs, description in test_cases:
        output = neuron.activate(inputs)
        weighted_sum = np.dot(inputs, neuron.weights)
        print(f"  {description:10s}  I={inputs}  Sum={weighted_sum}  T={neuron.threshold}  → {'FIRE' if output else 'silent'}")


def demo_logic_gates():
    """
    Demonstrate all standard Boolean logic gates using M-P neurons.

    McCulloch & Pitts showed that by manipulating weights and thresholds,
    a single neuron can implement AND, OR, NOT, NOR gates. This was their
    key insight: neurons are logic gates.
    """
    print("\n" + "=" * 60)
    print("DEMO 2: Boolean Logic Gates — The Paper's Core Result")
    print("=" * 60)
    print("\nMcCulloch & Pitts proved that neurons can implement Boolean")
    print("functions by setting appropriate weights and thresholds.\n")

    gates = [AND_gate(), OR_gate(), NOT_gate(), NOR_gate(), NAND_gate()]
    for gate in gates:
        gate.print_truth_table()


def demo_xor_limitation():
    """
    Demonstrate XOR — the famous limitation of single-layer networks.

    A single M-P neuron CANNOT compute XOR because XOR is not linearly
    separable. This was later highlighted by Minsky & Papert (1969) for
    Perceptrons. The solution requires a NETWORK of multiple neurons
    arranged in layers — foreshadowing multi-layer neural networks.
    """
    print("=" * 60)
    print("DEMO 3: XOR — Why Single Neurons Are Not Enough")
    print("=" * 60)
    print("\nXOR is not linearly separable → no single M-P neuron can compute it.")
    print("Solution: a multi-layer network of M-P neurons.\n")
    print("Architecture: XOR(A,B) = OR(AND(A, NOT(B)), AND(NOT(A), B))")
    print("This requires 5 neurons across 3 layers.\n")

    xor = XOR_network()
    xor.print_truth_table()

    # Verify correctness
    expected = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    for a, b, exp in expected:
        result = xor.compute(np.array([a, b]))
        status = "✓" if result == exp else "✗"
        assert result == exp, f"XOR({a},{b}) = {result}, expected {exp}"
    print("All XOR test cases passed ✓")


def demo_network_composition():
    """
    Demonstrate building complex functions by composing M-P neurons.

    The paper's Theorems 1-2 proved that any Temporal Propositional Expression
    can be realized by an acyclic network of M-P neurons. Here we build a
    few examples to illustrate composition.
    """
    print("\n" + "=" * 60)
    print("DEMO 4: Network Composition — Toward Universal Computation")
    print("=" * 60)

    # --- Example: 3-input majority gate ---
    # Fires when 2 or more of 3 inputs are active
    print("\n--- 3-Input Majority Gate ---")
    print("Fires when at least 2 of 3 inputs are active.")
    majority = MPNeuron(weights=[1, 1, 1], threshold=2, name="MAJORITY-3")
    majority.print_truth_table()

    # --- Example: AND-3 gate ---
    print("--- 3-Input AND Gate ---")
    print("Fires only when all 3 inputs are active.")
    and3 = MPNeuron(weights=[1, 1, 1], threshold=3, name="AND-3")
    and3.print_truth_table()

    # --- Example: OR-3 gate ---
    print("--- 3-Input OR Gate ---")
    print("Fires when at least 1 of 3 inputs is active.")
    or3 = MPNeuron(weights=[1, 1, 1], threshold=1, name="OR-3")
    or3.print_truth_table()

    # --- Theorem connection ---
    print("These examples illustrate Theorems 1-2 of the paper:")
    print("  Any Boolean function over N binary inputs can be realized")
    print("  by a network of M-P neurons (potentially multi-layer).")
    print("\n  Furthermore, Theorems 8-10 proved that networks with LOOPS")
    print("  (feedback connections) are equivalent to first-order logic,")
    print("  and with a tape, are Turing-complete.")


def demo_inhibition():
    """
    Demonstrate the absolute inhibition property.

    From the paper (Theorem 4): if ANY inhibitory synapse receives an
    active input, the neuron is completely suppressed — it will NOT fire
    regardless of how strong the excitatory input is.

    This models biological inhibitory neurotransmitters (like GABA) that
    can veto neural firing.
    """
    print("\n" + "=" * 60)
    print("DEMO 5: Absolute Inhibition (Theorem 4)")
    print("=" * 60)
    print("\nIf any inhibitory input is active, the neuron CANNOT fire.")
    print("This models biological inhibitory neurotransmitters.\n")

    # Neuron with 2 excitatory + 1 inhibitory input, threshold=1
    neuron = MPNeuron(weights=[1, 1, -1], threshold=1, name="2-excit-1-inhib")
    print(f"{neuron}\n")

    test_cases = [
        [0, 0, 0],  # no input → silent
        [1, 0, 0],  # excitatory only → fire
        [1, 1, 0],  # both excitatory → fire
        [1, 1, 1],  # excitatory + inhibitory → VETOED
        [0, 0, 1],  # inhibitory only → vetoed
    ]

    for inputs in test_cases:
        output = neuron.activate(inputs)
        inhib_active = inputs[2] == 1
        print(f"  I={inputs}  → {'FIRE' if output else 'silent'}"
              f"{'  ← inhibitory veto!' if inhib_active and not output else ''}")


# =============================================================================
# 5. MAIN — Run all demonstrations
# =============================================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  McCulloch & Pitts (1943) — Implementation             ║")
    print("║  'A Logical Calculus of the Ideas Immanent in           ║")
    print("║   Nervous Activity'                                     ║")
    print("║                                                         ║")
    print("║  The first mathematical model of artificial neurons.    ║")
    print("║  Foundation of neural computation & automata theory.    ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demo_single_neuron()
    demo_logic_gates()
    demo_xor_limitation()
    demo_network_composition()
    demo_inhibition()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Key results demonstrated:
  1. A single M-P neuron implements AND, OR, NOT, NOR, NAND gates
  2. XOR requires a multi-layer network (not linearly separable)
  3. Absolute inhibition can veto any excitatory input
  4. Arbitrary Boolean functions are realizable via neuron networks
  5. With feedback loops → first-order logic (Theorems 8-10)
  6. With a tape → Turing-complete computation

Limitations of the M-P model:
  • Binary I/O only (no continuous values)
  • No learning algorithm (weights are fixed)
  • Manual parameter tuning required
  → These limitations were addressed by Rosenblatt (1958) and
    Rumelhart, Hinton & Williams (1986)
""")
