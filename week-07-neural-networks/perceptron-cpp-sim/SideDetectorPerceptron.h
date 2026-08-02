/**
 * @file SideDetectorPerceptron.h
 * @brief A single-layer perceptron that learns to classify which side of a
 *        20x20 card a rectangular shape is on (left vs right).
 *
 * Architecture (following Rosenblatt 1958):
 *   - 400 input units (one per pixel) — these are the S-points / retina.
 *   - 400 learnable weights — analogous to A-unit values (V).
 *   - 1 output unit with a sign-based activation:
 *       sum < 0  →  "left"
 *       sum >= 0 →  "right"
 *
 * Training uses the perceptron learning rule with a small learning rate
 * (delta = ±0.1) to incrementally adjust weights on misclassified examples.
 */

#pragma once
#include <vector>
#include "RosenblattCard.h"

using namespace std;

class SideDetectorPerceptron
{
public:
    /// Initializes 400 weights to random values in [-1.0, +1.0].
    SideDetectorPerceptron();

    /// Trains on a set of labeled cards using the perceptron learning rule.
    /// On each misclassified card, active-pixel weights are nudged by ±0.1.
    /// @param cards  Reference to a vector of labeled RosenblattCards.
    void train(vector < RosenblattCard >& cards);

    /// Predicts "left" or "right" by computing the dot product of pixel
    /// values and weights, then applying a sign activation.
    /// @param card  The card to classify.
    /// @return "left" if weighted sum < 0, "right" otherwise.
    string predict(RosenblattCard& card);

    /// Prints the 400 weights in a 20x20 grid (row-major) to stdout.
    void printWeights();

private:
    vector < double > weights;  ///< 400 learnable weights (one per pixel)
};
