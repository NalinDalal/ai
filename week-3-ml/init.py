"""
init.py - Modular Neural Network Initialization and Training
"""
from initializations import initialize_parameters
from model import NeuralNetwork
from datasets import load_dataset

def main():
    # Load data
    train_X, train_Y, test_X, test_Y = load_dataset()

    # Choose initialization and regularization
    init_method = "he"  # Options: "zeros", "random", "he"
    lambd = 0.7         # L2 regularization strength (0.0 to disable)
    keep_prob = 0.8     # Dropout keep probability (1.0 to disable)
    layers_dims = [train_X.shape[0], 10, 5, 1]
    parameters = initialize_parameters(layers_dims, method=init_method)

    # Build and train model with regularization
    nn = NeuralNetwork(layers_dims, parameters, lambd=lambd, keep_prob=keep_prob)
    nn.train(train_X, train_Y, num_iterations=15000, learning_rate=0.01)

    # Evaluate
    print("Train accuracy:", nn.evaluate(train_X, train_Y))
    print("Test accuracy:", nn.evaluate(test_X, test_Y))

if __name__ == "__main__":
    main()
