# Rosenblatt Perceptron — C++ Simulation

> Recreating Frank Rosenblatt's 1958 perceptron experiment in C++.
> Based on Mark Mahoney's walkthrough. Inspiration from Brian Christian's *The Alignment Problem*.

---

## Background

In 1958, Frank Rosenblatt demonstrated one of the first examples of machine learning. He had a deck of cards, each with a solid shape on the **left** or **right** side. A camera captured a **20x20 pixel** image of each card, and the perceptron learned to predict which side the shape was on — without any hand-coded algorithm.

---

## Project Structure

| File | Purpose |
|------|---------|
| RosenblattCard.h/.cpp | Represents a 20x20 card with a rectangular shape |
| SideDetectorPerceptron.h/.cpp | The perceptron: 400 weights, training, and prediction |
| main.cpp | Data generation, training on 50k cards, testing on 500 |

---

## How It Works

### 1. The Card (RosenblattCard)

Each card is a **20x20 grid** stored as a flat array of 400 integers:
- 1 = active pixel (part of the shape)
- 0 = inactive pixel (background)

Conversion: index = row * 20 + col

A rectangle is placed at a random position via addRectangle(x, y, width, height). The label (left or right) is determined by counting how many shape pixels fall on each side of the center line (column 10).

### 2. The Non-AI Baseline (findSide)

Before using the perceptron, the code includes a simple algorithmic baseline:
- Iterate through all 400 pixels
- Count 1s in columns 0-9 (left) vs columns 10-19 (right)
- Whichever side has more active pixels wins

This achieves **100% accuracy** — but it only works because the problem is trivially separable. ML shines when problems are not this simple.

### 3. The Perceptron (SideDetectorPerceptron)

The perceptron has **400 weights** (one per pixel), initialized to random values in [-1.0, +1.0].

#### Prediction

    sum = sum of (pixel[i] * weight[i])  for i = 0..399

    if sum < 0  -> left
    if sum >= 0 -> right

**Intuition:** For this to work, weights on the left side (cols 0-9) should become negative, and weights on the right side (cols 10-19) should become positive. Then a shape on the left produces a negative sum, and vice versa.

#### Training (Perceptron Learning Rule)

For each labeled card:
1. Make a prediction using current weights
2. If **correct** -> do nothing
3. If **wrong** -> adjust weights at active pixels:
   - Predicted left but actual right -> weight[i] += 0.1 (nudge positive)
   - Predicted right but actual left -> weight[i] -= 0.1 (nudge negative)

The small delta (+-0.1) avoids the full-blast problem — slowly building confidence rather than overwriting with +-1.0 on every mistake.

### 4. Data Generation (loadData)

Randomly generates cards with:
- Random position: x in [0,17], y in [0,17]
- Random size: width in {3,5,7,9} (odd to avoid even splits), height in [2,8]

---

## Results

| Training Size | Test Size | Accuracy |
|---------------|-----------|----------|
| 500           | 5         | ~80%     |
| 50,000        | 500       | **100%** |

After training on 50,000 cards, the weight matrix shows a clear pattern:
- **Negative weights** cluster on the left half
- **Positive weights** cluster on the right half
- Weights near the **center columns (7-12)** are noisier, since shapes often straddle the middle

---

## Key Insights

1. **Zeroing problem:** Initializing all weights to 0.0 means only incorrect predictions update weights — pixels that are never part of a mistake stay at 0 forever. Random initialization solves this.

2. **Learning rate matters:** Full-blast updates (+-1.0) cause recent examples to dominate. Incremental updates (+-0.1) let the perceptron accumulate evidence across many examples.

3. **Center ambiguity:** The only incorrect predictions happen with shapes near the center line. Edge shapes are always predicted correctly because those weights converge quickly.

4. **Single neuron limitation:** This perceptron is a single neuron with 400 inputs. Stacking many neurons in layers (neural networks) enables learning far more complex patterns — that insight came later.

---

## Build and Run

    g++ -std=c++17 -o perceptron main.cpp RosenblattCard.cpp SideDetectorPerceptron.cpp
    ./perceptron

---

## Challenges

1. **Count training errors:** How many incorrect predictions occur during the 50k training pass?
2. **Convergence point:** At what training size does the perceptron stop making mistakes? 500? 5,000? 50,000?
3. **Port it:** Reimplement in Python, Rust, or your language of choice.
