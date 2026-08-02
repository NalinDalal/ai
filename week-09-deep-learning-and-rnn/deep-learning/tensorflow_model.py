"""
tensorflow_model.py - Neural Network using TensorFlow
Transition from NumPy to TensorFlow framework
"""

import numpy as np
import tensorflow as tf


def linear_function():
    """
    Implements a linear function:
    Y = WX + b

    Returns:
    result -- Y = WX + b
    """
    np.random.seed(1)

    X = tf.constant(np.random.randn(3, 1), name="X")
    W = tf.constant(np.random.randn(4, 3), name="W")
    b = tf.constant(np.random.randn(4, 1), name="b")
    Y = tf.add(tf.matmul(W, X), b)

    return Y


def sigmoid(z):
    """
    Computes the sigmoid of z

    Arguments:
    z -- input value, scalar or vector

    Returns:
    a -- (tf.float32) the sigmoid of z
    """
    z = tf.cast(z, tf.float32)
    a = tf.keras.activations.sigmoid(z)
    return a


def one_hot_matrix(label, depth=6):
    """
    Computes the one hot encoding for a single label

    Arguments:
    label -- (int) Categorical labels
    depth -- (int) Number of different classes that label can take

    Returns:
    one_hot -- tf.Tensor A single-column matrix with the one hot encoding.
    """
    one_hot = tf.one_hot(label, depth, axis=0)
    one_hot = tf.reshape(one_hot, [depth, 1])
    return one_hot


def initialize_parameters():
    """
    Initializes parameters to build a neural network with TensorFlow. The shapes are:
        W1 : [25, 12288]
        b1 : [25, 1]
        W2 : [12, 25]
        b2 : [12, 1]
        W3 : [6, 12]
        b3 : [6, 1]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    initializer = tf.keras.initializers.GlorotNormal(seed=1)

    W1 = tf.Variable(initializer(shape=(25, 12288)), name="W1")
    b1 = tf.Variable(initializer(shape=(25, 1)), name="b1")
    W2 = tf.Variable(initializer(shape=(12, 25)), name="W2")
    b2 = tf.Variable(initializer(shape=(12, 1)), name="b2")
    W3 = tf.Variable(initializer(shape=(6, 12)), name="W3")
    b3 = tf.Variable(initializer(shape=(6, 1)), name="b3")

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

    return parameters


@tf.function
def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.keras.activations.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.keras.activations.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)

    return Z3


@tf.function
def compute_cost(logits, labels):
    """
    Computes the cost

    Arguments:
    logits -- output of forward propagation, of shape (6, number of examples)
    labels -- "true" labels vector, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """
    cost = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(labels, logits, from_logits=True)
    )
    return cost


def model(
    X_train,
    Y_train,
    X_test,
    Y_test,
    learning_rate=0.0001,
    num_epochs=1500,
    minibatch_size=32,
    print_cost=True,
):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X_train -- training set
    Y_train -- training labels
    X_test -- test set
    Y_test -- test labels
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model
    """
    costs = []

    parameters = initialize_parameters()

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    optimizer = tf.keras.optimizers.SGD(learning_rate)

    X_train = X_train.batch(minibatch_size, drop_remainder=True).prefetch(8)
    Y_train = Y_train.batch(minibatch_size, drop_remainder=True).prefetch(8)

    for epoch in range(num_epochs):
        epoch_cost = 0.0

        for minibatch_X, minibatch_Y in zip(X_train, Y_train):
            with tf.GradientTape() as tape:
                Z3 = forward_propagation(minibatch_X, parameters)
                minibatch_cost = compute_cost(Z3, minibatch_Y)

            trainable_variables = [W1, b1, W2, b2, W3, b3]
            grads = tape.gradient(minibatch_cost, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))

            epoch_cost += minibatch_cost / minibatch_size

        if print_cost and epoch % 10 == 0:
            print(f"Cost after epoch {epoch}: {epoch_cost}")
        if print_cost and epoch % 5 == 0:
            costs.append(epoch_cost)

    print("Parameters have been trained!")

    return parameters


def create_dataset(X, Y, batch_size=32, shuffle=True):
    """
    Create TensorFlow dataset from numpy arrays

    Arguments:
    X -- input data
    Y -- labels
    batch_size -- batch size
    shuffle -- whether to shuffle

    Returns:
    dataset -- TensorFlow Dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(8)
    return dataset


def normalize(image):
    """
    Transform an image into a tensor of shape (flattened, 1) and normalize

    Arguments:
    image -- Tensor

    Returns:
    result -- Transformed tensor
    """
    image = tf.cast(image, tf.float32) / 256.0
    image = tf.reshape(image, [-1, 1])
    return image
