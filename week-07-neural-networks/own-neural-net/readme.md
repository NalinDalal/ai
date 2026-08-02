# Neural Network from Scratch
neurons have some input, their corresponding weights and then you add some bias to it
the output is:
```python
inputs = [ 1 , 2 , 3 , 2.5 ]
weights = [[ 0.2 , 0.8 , -0.5 ,1],
[0.5,
- 0.91 , 0.26 ,
- 0.5 ],
[-0.26,- 0.27 , 0.17 , 0.87 ]]

biases = [ 2 , 3 , 0.5 ]
# Output of current layer
layer_outputs = []

# For each neuron
for neuron_weights, neuron_bias in zip (weights, biases):
    # Zeroed output of given neuron
    neuron_output = 0
    # For each input and weight to the neuron
    for n_input, weight in zip (inputs, neuron_weights):
        # Multiply this input by associated weight
        # and add to the neuron’s output variable
        neuron_output += n_input * weight
    # Add bias
    neuron_output += neuron_bias
    # Put neuron’s result to the layer’s output list
    layer_outputs.append(neuron_output)
print (layer_outputs)
```

then you can arrange multiple neurons on a sort of stack
so called neural network

fully connected neural network — every
neuron in the current layer has connections to every neuron from the previous layer

## Tensors, Arrays and Vectors
u know array, then you know about matrix, and 3d array
A tensor object is an object that can be represented as an array.

then u know about dot product of vector
```python
a = [ 1 , 2 , 3 ]
b = [ 2 , 3 , 4 ]
dot_product = a[ 0 ] * b[ 0 ] + a[ 1 ] * b[ 1 ] + a[ 2 ] * b[ 2 ]
print (dot_product)
```

use numpy to calculate the dot product
```python
import numpy as np
inputs = [ 1.0 , 2.0 , 3.0 , 2.5 ]
weights = [ 0.2 , 0.8 ,- 0.5 , 1.0 ]
bias = 2.0
outputs = np.dot(weights, inputs) + bias
print (outputs)
```

Layer of Neuron with NumPy
```python
import numpy as np
inputs = [ 1.0 , 2.0 , 3.0 , 2.5 ]
weights = [[ 0.2 , 0.8 ,- 0.5 , 1 ],
[ 0.5 ,- 0.91 , 0.26 ,- 0.5 ],
[ - 0.26 ,- 0.27 , 0.17 , 0.87 ]]
biases = [ 2.0 , 3.0 , 0.5 ]
layer_outputs = np.dot(weights, inputs) + biases
print (layer_outputs)

#array([ 4.8 1.21 2.385 ])
```

then u know how to find matrix multiplication

## Adding Layers
2/more hidden layer complicate the network
say Input layer with 4 features into a hidden layer with 3 neurons.
```python
import numpy as np
inputs = [[ 1 , 2 , 3 , 2.5 ], [ 2.0 , 5.0 ,-1.0,2.0],[ - 1.5 , 2.7 , 3.3 , -0.8 ]]
weights = [[ 0.2 , 0.8 ,- 0.5 , 1 ],
[ 0.5 ,- 0.91 , 0.26 ,- 0.5 ],
[ - 0.26 ,- 0.27 , 0.17 , 0.87 ]]
biases = [ 2 , 3 , 0.5 ]
weights2 = [[ 0.1 ,- 0.14 , 0.5 ],
[ - 0.5 , 0.12 ,- 0.33 ],
[ - 0.44 , 0.73 ,- 0.13 ]]
biases2 = [ - 1 , 2 ,- 0.5 ]


layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print (layer2_outputs)
```
```
array([[ 0.5031 - 1.04185 - 2.03875 ],
[ 0.2434 - 2.7332 - 5.7633 ],
[ - 0.99314 1.41254 - 0.35655 ]])
```

creating a dataset, that is non-linear
```python
from nnfs.datasets import spiral_data
import numpy as np
import nnfs
import matplotlib.pyplot as plt

nnfs.init() #sets the random seed to 0 (by the default), creates a float32 dtype default, and overrides the original dot product from NumPy

X, y = spiral_data( samples = 100 , classes = 3 )   #create a dataset with as many classes as we want.
plt.scatter(X[:, 0 ],X[:, 1 ])
plt.show()

#adding colour to chart
plt.scatter(X[:, 0 ], X[:, 1 ], c = y, cmap = 'brg' )
plt.show()
```

## Dense Layer Class
```python
class Layer_Dense :
    def __init__ ( self , n_inputs , n_neurons ):
        # Initialize weights and biases(randomly)
        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons) 
        self.biases = np.zeros((1 , n_neurons))
        pass # using pass statement as a placeholder

    # Forward pass
    def forward ( self , inputs ):
        # Calculate output values from inputs, weights and biases
        pass # using pass statement as a placeholder

# Create dataset
X, y = spiral_data( samples = 100 , classes = 3 )

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense( 2 , 3 )

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Let's see output of the first few samples:
print (dense1.output[: 5 ])
```

# Activation Functions
for a neural network to fit a nonlinear function, 
we need it to contain two or more hidden layers, and
we need those hidden layers to use a nonlinear activation function.

u connect neurons to justify the graphs

**ReLU**
relu: if current value is greater than 0 then set to 1, else to 0
```python
inputs = [ 0 , 2 ,- 1 , 3.3 ,- 2.7 , 1.1 , 2.2 ,- 100 ]

output = []
for i in inputs:
    if i > 0 :
        output.append(i)
    else :
    output.append(0)
print (output)
```

ReLU in this code is a loop where we’re checking if the
current value is greater than 0. If it is, we’re appending it to the output list, and if it’s not, we’re
appending 0. This can be written more simply, as we just need to take the largest of two values: 0
or neuron value

```python
inputs = [ 0 , 2 ,- 1 , 3.3 ,- 2.7 , 1.1 , 2.2 ,- 100 ]

output = []
for i in inputs:
    output.append(max(0,i))
print (output)
```

applying to our neural network:
```python
# ReLU activation
class Activation_ReLU :
    # Forward pass
    def forward ( self , inputs ):
        # Calculate output values from input
        self.output = np.maximum( 0 , inputs)

# Create dataset
X, y = spiral_data( samples = 100 , classes = 3 )

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense( 2 , 3 )
# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()
    
# Make a forward pass of our training data through this layer
dense1.forward(X)
    
# Forward pass through activation func.
# Takes in output from previous layer
activation1.forward(dense1.output)

# Let's see output of the first few samples:
print (activation1.output[: 5 ])
```

**SoftMax Function:**
activation function meant for classification
function represents confidence scores for each class and will add up to 1.
For example, 
if our network has a confidence distribution for two classes: [0.45, 0.55] , 
the prediction is the 2nd class, but the confidence in this prediction isn’t very high. 
Maybe our program would not act in this case since it’s not very confident.

Here's the function for the **Softmax**:

$$
S_{i,j} = \frac{e^{z_{i,j}}}{\sum_{l=1}^{L} e^{z_{i,l}}}
$$

```python
# Values from the previous output when we described
# what a neural network is
layer_outputs = [ 4.8 , 1.21 , 2.385 ]
# e - mathematical constant, we use E here to match a common coding
# style where constants are uppercased
E = 2.71828182846 # you can also use math.e
# For each value in a vector, calculate the exponential value
exp_values = []
for output in layer_outputs:
    exp_values.append(E ** output) # ** - power operator in Python
print ( 'exponentiated values:' )
print (exp_values)

>>>
exponentiated values:
[ 121.51041751893969 , 3.3534846525504487 , 10.85906266492961 ]
```

exponential function is a monotonic function. This means that, with higher input values,
outputs are also higher, so we won’t change the predicted class after applying it while making
sure that we get non-negative values. It also adds stability to the result as the normalized
exponentiation is more about the difference between numbers than their magnitudes.

```python
# Now normalize values
norm_base = sum (exp_values) # We sum all values
norm_values = []
for value in exp_values:
    norm_values.append(value / norm_base)
print ( 'Normalized exponentiated values:' )
print (norm_values)

print ( 'Sum of normalized values:', sum (norm_values))
```

```
>>>
Normalized exponentiated values:
[ 0.8952826639573506 , 0.024708306782070668 , 0.08000902926057876 ]
Sum of normalized values: 1.0
```

same via numpy:
```python
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

# Dense layer
class Layer_Dense :
    # Layer initialization
    def __init__ ( self , n_inputs , n_neurons ):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros(( 1 , n_neurons))

    # Forward pass
    def forward ( self , inputs ):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

# ReLU activation
class Activation_ReLU :
    # Forward pass
    def forward ( self , inputs ):
        # Calculate output values from inputs
        self.output = np.maximum( 0 , inputs)

# Softmax activation
class Activation_Softmax :
    # Forward pass
    def forward ( self , inputs ):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis = 1 ,keepdims = True ))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis = 1 ,keepdims = True )
        self.output = probabilities

# Common loss class
# calculating over-all loss
class Loss :
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate ( self , output , y ):

        # Calculate sample losses
        sample_losses = self.forward(output, y)
        
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
    
        # Return loss
        return data_loss

# Cross-entropy loss
class Loss_CategoricalCrossentropy ( Loss ):
    # Forward pass
    def forward ( self , y_pred , y_true ):
        # Number of samples in a batch
        samples = len (y_pred)
        
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7 , 1 - 1e-7 )

        # Probabilities for target values -
        # only if categorical labels
        if len (y_true.shape) == 1 :
            correct_confidences = y_pred_clipped[
                range (samples),
                y_true
            ]
        
        # Mask values - only for one-hot encoded labels
        elif len (y_true.shape) == 2 :
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis = 1
            )
        
        # Losses
        negative_log_likelihoods = - np.log(correct_confidences)
        return negative_log_likelihoods

# Create dataset
X, y = spiral_data( samples = 100 , classes = 3 )

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense( 2 , 3 )

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense( 3 , 3 )

# Create Softmax activation (to be used with Dense layer):
activation2 = Activation_Softmax()

# Make a forward pass of our training data through this layer
dense1.forward(X)

# Make a forward pass through activation function
# it takes the output of first dense layer here
activation1.forward(dense1.output)

# Make a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Make a forward pass through activation function
# it takes the output of second dense layer here
activation2.forward(dense2.output)

# Let's see output of the first few samples:
print (activation2.output[: 5 ])
```

```
>>>
[[ 0.33333334 0.33333334 0.33333334 ]
[ 0.33333316 0.3333332 0.33333364 ]
[ 0.33333287 0.3333329 0.33333418 ]
[ 0.3333326 0.33333263 0.33333477 ]
[ 0.33333233 0.3333324 0.33333528 ]]
```

---

# Calculating Network Error with Loss

find categorical cross entropy as a u know mean entropy or something


Where L i denotes sample loss value, i is the i-th sample in the set, j is the label/output index, y
denotes the target values, and y-hat denotes the predicted values

Categorical Cross-Entropy Loss:

$$
L_i = - \sum_j y_{i,j} \log(\hat{y}_{i,j})
$$

Where $L_i$ denotes sample loss value, $i$ is the i-th sample in the set, $j$ is the label/output index, $y$ denotes the target values, and $\hat{y}$ denotes the predicted values.

softmax output of [ 0.7 , 0.1 , 0.2 ] and targets of [ 1 , 0 , 0 ] , we can apply the
calculations as follows:
$$
L_i = - \sum_j y_{i,j} \log(\hat{y}_{i,j})
L_i= -(1*log(0.7)+0*log(0.1)+0*log(0.2))
    = -(-0.35667494393873245+0+0)
    = 0.35667494393873245
$$


```python
import math
# An example output from the output layer of the neural network
softmax_output = [ 0.7 , 0.1 , 0.2 ]

# Ground truth
target_output = [ 1 , 0 , 0 ]
loss = - (math.log(softmax_output[ 0 ])  * target_output[ 0 ]+
math.log(softmax_output[ 1 ]) * target_output[ 1 ] +
math.log(softmax_output[ 2 ]) * target_output[ 2 ] )

print (loss)
```

```
0.35667494393873245
```


## Accuracy Calculation
how often the largest confidence is the correct class
in terms of a fraction. 
use the argmax values from the softmax outputs and then compare
these to the targets

```python
import numpy as np
# Probabilities of 3 samples
softmax_outputs = np.array([[ 0.7 , 0.2 , 0.1 ],
[ 0.5 , 0.1 , 0.4 ],
[ 0.02 , 0.9 , 0.08 ]])

# Target (ground-truth) labels for 3 samples
class_targets = np.array([ 0 , 1 , 1 ])

# Calculate values along second axis (axis of index 1)
predictions = np.argmax(softmax_outputs, axis = 1 )
# If targets are one-hot encoded - convert them
if len (class_targets.shape) == 2 :
    class_targets = np.argmax(class_targets, axis = 1 )
# True evaluates to 1; False to 0
accuracy = np.mean(predictions == class_targets)

print ( 'acc1:', accuracy)

# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(activation2.output, axis = 1 )
if len (y.shape) == 2 :
    y = np.argmax(y, axis = 1 )
accuracy2 = np.mean(predictions == y)

# Print accuracy
print ( 'acc2:' , accuracy2)
```

```
acc1: 0.6666666666666666
acc2: 0.34
```

# Optimisations
randomly changing the weights, checking the loss, and
repeating this until happy with the lowest loss found.
```python

# Create dataset
X, y = vertical_data( samples = 100 , classes = 3 )

# Create model
dense1 = Layer_Dense( 2 , 3 ) # first dense layer, 2 inputs
activation1 = Activation_ReLU()
dense2 = Layer_Dense( 3 , 3 ) # second dense layer, 3 inputs, 3 outputs
activation2 = Activation_Softmax()
# Create loss function
loss_function = Loss_CategoricalCrossentropy()

# Helper variables
lowest_loss = 9999999 # some initial value

best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()
```

iterate as many times as desired, pick random values for weights and biases, and
save the weights and biases if they generate the lowest-seen loss:

```python
for iteration in range ( 10000 ):
    # Generate a new set of weights for iteration
    dense1.weights = 0.05 * np.random.randn( 2 , 3 )
    dense1.biases = 0.05 * np.random.randn( 1 , 3 )
    dense2.weights = 0.05 * np.random.randn( 3 , 3 )
    dense2.biases = 0.05 * np.random.randn( 1 , 3 )
    
    # Perform a forward pass of the training data through this layer
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    # Perform a forward pass through activation function
    # it takes the output of second dense layer here and returns loss
    loss = loss_function.calculate(activation2.output, y)

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(activation2.output, axis = 1 )
    accuracy = np.mean(predictions == y)
    
    # If loss is smaller - print and save weights and biases aside
    if loss < lowest_loss:
        print ( 'New set of weights found, iteration:', iteration,
            'loss:', loss,'acc:', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
```

# Derivatives

consider a simple function : 
y=f(x)
f(x)=2*x

plotting x v/s y graph:
```python
import matplotlib.pyplot as plt
import numpy as np
def f ( x ):
    return 2 * x

x = np.array( range ( 5 ))
y = f(x)
print (x)   #[ 0 1 2 3 4 ]
print (y)   #[ 0 2 4 6 8 ]

plt.plot(x, y)
plt.show()
```

slope: $\frac{y2-y1}{x2-x1}$

printing this slope:
```py
print ((y[ 1 ] - y[ 0 ]) / (x[ 1 ] - x[ 0 ]))
```

we can calulate similarly derivatives for other functions

**Numerical Derivative**
calculating the
slope of the tangent line using two infinitely close points
```python
import matplotlib.pyplot as plt
import numpy as np
def f ( x ):
    return 2 * x ** 2
# np.arange(start, stop, step) to give us smoother line
x = np.arange( 0 , 5 , 0.001 )
y = f(x)
plt.plot(x, y)
plt.show()
```

hence you can plot various functions

**Derivatives**
U know how to find derivatives of functions

The derivative of a constant equals 0 (m is a constant in this case, as it’s not a parameter that we are deriving with respect to, which is x in this example):

$$
\frac{d}{dx} 1 = 0
$$

$$
\frac{d}{dx} m = 0
$$

The derivative of $x$ equals 1:

$$
\frac{d}{dx} x = 1
$$

The derivative of a linear function equals its slope:

$$
\frac{d}{dx} (mx + b) = m
$$

Rules:

The derivative of a constant multiple of the function equals the constant multiple of the function’s derivative:

$$
\frac{d}{dx} [k \cdot f(x)] = k \cdot \frac{d}{dx} f(x)
$$

The derivative of a sum of functions equals the sum of their derivatives:

$$
\frac{d}{dx} [f(x) + g(x)] = \frac{d}{dx} f(x) + \frac{d}{dx} g(x) = f'(x) + g'(x)
$$

The same concept applies to subtraction:

$$
\frac{d}{dx} [f(x) - g(x)] = \frac{d}{dx} f(x) - \frac{d}{dx} g(x) = f'(x) - g'(x)
$$

The derivative of an exponentiation:

$$
\frac{d}{dx} x^n = n \cdot x^{n-1}
$$

---

# Gradients, Partial Derivatives, and the Chain Rule

**Gradient:**
- The gradient is a vector of partial derivatives. It points in the direction of the greatest rate of increase of a function.
- For a function $f(x, y)$, the gradient is:
  $$
  \nabla f = \left( \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right)
  $$

**Partial Derivative:**
- A partial derivative measures how a function changes as only one variable changes, keeping others constant.
- For $f(x, y)$:
  $$
  \frac{\partial f}{\partial x}, \quad \frac{\partial f}{\partial y}
  $$

**Chain Rule:**
- The chain rule is used to compute the derivative of composite functions.
- If $y = f(g(x))$, then:
  $$
  \frac{dy}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}
  $$
- For multivariable functions:
  $$
  \frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x}
  $$

**Example:**
If $z = f(y)$ and $y = g(x)$, then:
$$
\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}
$$

---

# BackPropogation

u code how actually your neural networks sit with each other, it's like u create a map via draeing then implement that in code
```python
# Forward pass
x = [ 1.0 , - 2.0 , 3.0 ] # input values
w = [ - 3.0 ,- 1.0 , 2.0 ] # weights
b = 1.0 # bias


# Multiplying inputs by weights
xw0 = x[ 0 ] * w[ 0 ]
xw1 = x[ 1 ] * w[ 1 ]
xw2 = x[ 2 ] * w[ 2 ]

# Adding weighted inputs and a bias
z = xw0 + xw1 + xw2 + b

# ReLU activation function
y = max (z, 0 )

# Backward pass
# The derivative from the next layer
dvalue = 1.0

# Derivative of ReLU and the chain rule
drelu_dz = dvalue * ( 1. if z > 0 else 0. )
print (drelu_dz)

# Partial derivatives of the multiplication, the chain rule
dsum_dxw0 = 1

dsum_dxw1 = 1

dsum_dxw2 = 1

dsum_db = 1

drelu_dxw0 = drelu_dz * dsum_dxw0
drelu_dxw1 = drelu_dz * dsum_dxw1
drelu_dxw2 = drelu_dz * dsum_dxw2
drelu_db = drelu_dz * dsum_db
print (drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)

# Partial derivatives of the multiplication, the chain rule
dmul_dx0 = w[ 0 ]
drelu_dx0 = drelu_dxw0 * dmul_dx0
print (drelu_dx0)
```

**Categorical Cross-Entropy loss derivative**

$$
L_i = -\log(\hat{y}_{i,k})
$$
where $k$ is the index of the "true" probability.

Full solution (derivative):

$$
\frac{\partial L_i}{\partial \hat{y}_{i,j}} = \frac{\partial}{\partial \hat{y}_{i,j}}\left[-\sum_j y_{i,j} \log(\hat{y}_{i,j})\right] = -\sum_j y_{i,j} \cdot \frac{1}{\hat{y}_{i,j}} \cdot \frac{\partial}{\partial \hat{y}_{i,j}} \hat{y}_{i,j}
$$
$$
= -\sum_j y_{i,j} \cdot \frac{1}{\hat{y}_{i,j}} \cdot 1 = -\sum_j \frac{y_{i,j}}{\hat{y}_{i,j}}
$$

```python
# Cross-entropy loss
class Loss_CategoricalCrossentropy ( Loss ):
    ...
    # Backward pass
    def backward ( self , dvalues , y_true ):
        # Number of samples
        samples = len (dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len (dvalues[ 0 ])
        
        # If labels are sparse, turn them into one-hot vector
        if len (y_true.shape) == 1 :
            y_true = np.eye(labels)[y_true]
        
        # Calculate gradient
        self.dinputs = - y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples
```

**Softmax activation derivative**
$$
\frac{\partial S_{i,j}}{\partial z_{i,k}} = S_{i,j} \cdot (\delta_{j,k} - S_{i,k}) = S_{i,j} \delta_{j,k} - S_{i,j} S_{i,k}
$$

Where $\delta_{j,k}$ is the Kronecker delta (1 if $j=k$, 0 otherwise).
```python
# Softmax activation
class Activation_Softmax :
    ...
    # Backward pass
    def backward ( self , dvalues ):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate ( zip (self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape( - 1 , 1 )
            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)
```

**Softmax classifier - combined Softmax activation and cross-entropy loss for faster backward step**
```python
class Activation_Softmax_Loss_CategoricalCrossentropy ():
    # Creates activation and loss function objects
    def __init__ ( self ):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
    
    # Forward pass
    def forward ( self , inputs , y_true ):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    
    # Backward pass
    def backward ( self , dvalues , y_true ):
        # Number of samples
        samples = len (dvalues)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len (y_true.shape) == 2 :
            y_true = np.argmax(y_true, axis = 1 )

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[ range (samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples
```

# Optimisers


### Stochastic Gradient Descent (SGD)
Stochastic Gradient Descent updates the model parameters using the gradient of the loss function with respect to the parameters. It uses a small batch of data (or a single data point) to compute the gradient, making it computationally efficient but noisy.

**Function:**
```python
# SGD update rule
w = w - learning_rate * gradient
```

### Learning Rate
The learning rate controls the step size during the parameter update. A high learning rate may overshoot the optimal solution, while a low learning rate may result in slow convergence.

### Learning Rate Decay
Learning rate decay reduces the learning rate over time to ensure convergence. This helps the model settle into a minimum by taking smaller steps as training progresses.

**Function:**
```python
# Learning rate decay
learning_rate = initial_lr / (1 + decay_rate * epoch)
```

### Stochastic Gradient Descent with Momentum
Momentum accelerates SGD by adding a fraction of the previous update to the current update. This helps navigate ravines and reduces oscillations.

**Function:**
```python
# SGD with Momentum
velocity = momentum * velocity - learning_rate * gradient
w = w + velocity
```

### AdaGrad
AdaGrad adapts the learning rate for each parameter based on the historical gradients. It performs well for sparse data but may lead to vanishing learning rates.

**Function:**
```python
# AdaGrad update rule
g = gradient
cache += g**2
w = w - (learning_rate / (sqrt(cache) + epsilon)) * g
```

### RMSProp
RMSProp modifies AdaGrad by introducing a decay factor to the historical gradients, preventing the learning rate from vanishing.

**Function:**
```python
# RMSProp update rule
g = gradient
cache = decay_rate * cache + (1 - decay_rate) * g**2
w = w - (learning_rate / (sqrt(cache) + epsilon)) * g
```

### Adam (Adaptive Moment Estimation)
Adam combines the benefits of Momentum and RMSProp. It maintains running averages of both the gradients and their squared values, making it robust and efficient.

**Function:**
```python
# Adam update rule
m = beta1 * m + (1 - beta1) * gradient
v = beta2 * v + (1 - beta2) * gradient**2
m_hat = m / (1 - beta1**t)
v_hat = v / (1 - beta2**t)
w = w - (learning_rate / (sqrt(v_hat) + epsilon)) * m_hat
```

# L1 Regularisation

L1 regularization adds a penalty proportional to the absolute value of the weights to the loss function. This encourages sparsity in the model by driving some weights to zero.

**Loss Function with L1 Regularization:**
```python
loss = original_loss + lambda_ * sum(abs(w))
```

## Backward Pass for L1 Regularization
During the backward pass, the gradient of the L1 penalty is added to the gradient of the loss function. The gradient of the L1 term is the sign of the weights.

**Backward Pass Implementation:**
```python
gradient += lambda_ * np.sign(w)
```

### Dropout

Dropout is a regularization technique used to prevent overfitting in neural networks. During training, it randomly sets a fraction of the neurons' outputs to zero, effectively removing them from the network for that iteration. This forces the network to learn more robust features.

**Implementation:**
```python
# Dropout during training
mask = (np.random.rand(*layer_output.shape) > dropout_rate)
layer_output *= mask
layer_output /= (1 - dropout_rate)  # Scale to maintain expected value
```

### Binary Logistic Regression

Binary Logistic Regression is a classification algorithm used to predict binary outcomes (e.g., 0 or 1). It models the probability of the positive class using the sigmoid function.

**Sigmoid Function:**
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

**Prediction:**
```python
# Logistic regression prediction
def predict(X, weights):
    z = np.dot(X, weights)
    return sigmoid(z)
```

---

### Binary Cross-Entropy Loss

Binary Cross-Entropy Loss measures the difference between the predicted probabilities and the true labels. It is commonly used as the loss function for binary classification tasks.

**Loss Function:**
```python
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # To avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```

**Gradient of the Loss:**
```python
def gradient(X, y_true, y_pred):
    return np.dot(X.T, (y_pred - y_true)) / y_true.size
```

### Regression and Linear Activation

Regression tasks involve predicting continuous values. In neural networks, the output layer for regression typically uses a linear activation function, which directly outputs the predicted value without applying any non-linearity.

**Linear Activation Function:**
```python
def linear_activation(z):
    return z
```

---

### Mean Squared Error (MSE) Loss

Mean Squared Error calculates the average squared difference between predicted and true values. It is commonly used for regression tasks.

**MSE Loss Function:**
```python
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)
```

**MSE Loss Derivative:**
```python
def mse_derivative(y_true, y_pred):
    return -2 * (y_true - y_pred) / y_true.size
```

---

### Mean Absolute Error (MAE) Loss

Mean Absolute Error calculates the average absolute difference between predicted and true values. It is less sensitive to outliers compared to MSE.

**MAE Loss Function:**
```python
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
```

**MAE Loss Derivative:**
```python
def mae_derivative(y_true, y_pred):
    return np.where(y_pred > y_true, 1, -1) / y_true.size
```

---

# Creating our Own Model

### Dense Layer

The dense layer is a fully connected layer that connects every input neuron to every output neuron. It includes weights and biases for each connection.

**Code Implementation:**
```python
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_L1=0, weight_regularizer_L2=0, bias_regularizer_L1=0, bias_regularizer_L2=0):
        # Initialize weights and biases
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Set regularization strength
        self.weight_regularizer_L1 = weight_regularizer_L1
        self.weight_regularizer_L2 = weight_regularizer_L2
        self.bias_regularizer_L1 = bias_regularizer_L1
        self.bias_regularizer_L2 = bias_regularizer_L2

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradients on regularization
        if self.weight_regularizer_L1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_L1 * dL1
        if self.weight_regularizer_L2 > 0:
            self.dweights += 2 * self.weight_regularizer_L2 * self.weights
        if self.bias_regularizer_L1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_L1 * dL1
        if self.bias_regularizer_L2 > 0:
            self.dbiases += 2 * self.bias_regularizer_L2 * self.biases
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
```

---

### Dropout Layer

Dropout is a regularization technique that randomly disables a fraction of neurons during training to prevent overfitting.

**Code Implementation:**
```python
class Layer_Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask
```

---

### Activation Functions

#### ReLU Activation
ReLU (Rectified Linear Unit) sets all negative values to zero, introducing non-linearity to the model.

**Code Implementation:**
```python
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
```

#### Sigmoid Activation
The sigmoid function maps input values to a range between 0 and 1, making it suitable for binary classification.

**Code Implementation:**
```python
class Activation_Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output
```

#### Linear Activation
Linear activation is used for regression tasks, where the output is a continuous value.

**Code Implementation:**
```python
class Activation_Linear:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
```

---

### Optimizers

#### SGD Optimizer
Stochastic Gradient Descent updates weights using the gradient of the loss function.

**Code Implementation:**
```python
class Optimizer_SGD:
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1
```

```python
# Model class
class Model:
    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self):
        self.input_layer = Layer_Input()
        layer_count = len(self.layers)
        self.trainable_layers = []
        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
        self.loss.remember_trainable_layers(self.trainable_layers)
        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossentropy):
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()

    def train(self, X, y, *, validation_data=None, epochs=1, print_every=1):
        self.accuracy.init(y)
        for epoch in range(1, epochs + 1):
            output = self.forward(X, training=True)
            data_loss, regularization_loss = self.loss.calculate(output, y, include_regularization=True)
            loss = data_loss + regularization_loss
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)
            self.backward(output, y)
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()
            if not epoch % print_every:
                print(f'epoch: {epoch}, ' +
                      f'acc: {accuracy:.3f}, ' +
                      f'loss: {loss:.3f} (' +
                      f'data_loss: {data_loss:.3f}, ' +
                      f'reg_loss: {regularization_loss:.3f}), ' +
                      f'lr: {self.optimizer.current_learning_rate}')
            if validation_data is not None:
                X_val, y_val = validation_data
                output = self.forward(X_val, training=False)
                loss = self.loss.calculate(output, y_val)
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, y_val)
                print(f'validation, ' +
                      f'acc: {accuracy:.3f}, ' +
                      f'loss: {loss:.3f}')

    def forward(self, X, training):
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        return self.layers[-1].output

    def backward(self, output, y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return
        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

# Create dataset
X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

# Instantiate the model
model = Model()
model.add(Layer_Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(512, 3))
model.add(Activation_Softmax())

# Set loss, optimizer, and accuracy objects
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
    accuracy=Accuracy_Categorical()
)

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)
```

----

## Structure
- `layers.py`: Layer classes (Dense, Dropout, Input)
- `activations.py`: Activation functions (ReLU, Softmax, Sigmoid, Linear)
- `optimizers.py`: Optimizer classes (SGD, Adam, RMSprop, Adagrad)
- `losses.py`: Loss functions (Categorical, Binary, MSE, MAE)
- `accuracy.py`: Accuracy calculation classes
- `data.py`: Data loading/generation utilities
- `model.py`: Model class for building, training, and evaluating neural networks
- `main.py`: Entry point for building, training, and evaluating the model
