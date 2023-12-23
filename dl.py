import numpy as np


# Define the input data and expected output
X = np.array([[0, 0], [1, 1]])
y = np.array([[0], [1]])

# Normalize the input data
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_std[X_std == 0] = 1e-8  # Add a small constant to avoid divide-by-zero errors
X_norm = (X - X_mean) / X_std

# Normalize the output data
y_mean = np.mean(y, axis=0)
y_std = np.std(y, axis=0)
y_norm = (y - y_mean) / y_std

# Define the neural network architecture
input_layer_size = 2
hidden_layer_size = 20
output_layer_size = 1

# Initialize the weights and biases
W1 = np.random.randn(input_layer_size, hidden_layer_size)
b1 = np.random.randn(hidden_layer_size)
W2 = np.random.randn(hidden_layer_size, hidden_layer_size)
b2 = np.random.randn(hidden_layer_size)
W3 = np.random.randn(hidden_layer_size, output_layer_size)
b3 = np.random.randn(output_layer_size)


# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Define the derivative of the sigmoid function
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Set the learning rate and number of epochs
learning_rate = 0.01
num_epochs = 1_000

print("start training..")

# Train the neural network
for i in range(num_epochs):
    if i % 10_000 == 0 and i > 0:
        print((num_epochs - i) / 100, " left")

    # Forward pass
    z1 = np.dot(X_norm, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    z3 = np.dot(a2, W3) + b3
    y_pred = z3

    # Backward pass
    error = y_pred - y_norm
    delta3 = error
    delta2 = np.dot(delta3, W3.T) * sigmoid_prime(z2)
    delta1 = np.dot(delta2, W2.T) * sigmoid_prime(z1)

    # Update the weights and biases
    W3 -= learning_rate * np.dot(a2.T, delta3)
    b3 -= learning_rate * np.sum(delta3, axis=0)
    W2 -= learning_rate * np.dot(a1.T, delta2)
    b2 -= learning_rate * np.sum(delta2, axis=0)
    W1 -= learning_rate * np.dot(X_norm.T, delta1)
    b1 -= learning_rate * np.sum(delta1, axis=0)


# Test the neural network
test_input = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
test_input_norm = (test_input - X_mean) / X_std
z1 = np.dot(test_input_norm, W1) + b1
a1 = sigmoid(z1)
z2 = np.dot(a1, W2) + b2
a2 = sigmoid(z2)
z3 = np.dot(a2, W3) + b3
test_output_norm = z3
test_output = test_output_norm * y_std + y_mean
print(test_output)