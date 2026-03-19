"""
main.py
Entry point for building, training, and evaluating the neural network.
"""
import numpy as np
from data import spiral_data
from layers import Layer_Dense, Layer_Dropout
from activations import Activation_ReLU, Activation_Softmax
from optimizers import Optimizer_Adam
from losses import Loss_CategoricalCrossentropy
from accuracy import Accuracy_Categorical
from model import Model

# Generate dataset
X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

# Build model
model = Model()
model.add(Layer_Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(512, 3))
model.add(Activation_Softmax())

# Set loss, optimizer, and accuracy
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
    accuracy=Accuracy_Categorical()
)

# Finalize and train
model.finalize()
model.train(X, y, validation_data=(X_test, y_test), epochs=1000, print_every=100)
