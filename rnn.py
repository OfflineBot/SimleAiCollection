import numpy as np

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define the RNN class
class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.W_input_hidden = np.random.rand(input_size, hidden_size)
        self.W_hidden_output = np.random.rand(hidden_size, output_size)

        # Initialize biases
        self.b_hidden = np.zeros((1, hidden_size))
        self.b_output = np.zeros((1, output_size))

    def forward(self, X):
        # Forward pass
        self.hidden_state = sigmoid(np.dot(X, self.W_input_hidden) + self.b_hidden)
        self.output = sigmoid(np.dot(self.hidden_state, self.W_hidden_output) + self.b_output)
        return self.output

    def backward(self, X, y, learning_rate):
        # Backward pass
        error = y - self.output
        output_delta = error * sigmoid_derivative(self.output)

        hidden_error = output_delta.dot(self.W_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_state)

        # Update weights and biases
        self.W_hidden_output += self.hidden_state.T.dot(output_delta) * learning_rate
        self.W_input_hidden += X.T.dot(hidden_delta) * learning_rate

        self.b_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.b_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward and backward pass for each training example
            for i in range(len(X)):
                input_data = X[i]
                target = y[i]

                # Reshape input data to have a batch size of 1
                input_data = np.reshape(input_data, (1, -1))

                output = self.forward(input_data)
                self.backward(input_data, target, learning_rate)

            # Print the mean squared error at the end of each epoch
            mse = np.mean(np.square(y - self.predict(X)))
            print(f'Epoch {epoch + 1}/{epochs}, Mean Squared Error: {mse}')

    def predict(self, X):
        # Make predictions
        predictions = []
        for i in range(len(X)):
            input_data = X[i]
            input_data = np.reshape(input_data, (1, -1))
            predictions.append(self.forward(input_data))
        return np.array(predictions).squeeze()

# Custom data
# Assume you have a sequence of binary values as input
# and the reverse of that sequence as the output
# For example, input: [0, 1, 0, 1], output: [1, 0, 1, 0]
X_train = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1]])
y_train = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 0]])

# Input and output sizes
input_size = X_train.shape[1]
output_size = y_train.shape[1]

# Hyperparameters
hidden_size = 4
learning_rate = 0.1
epochs = 1000

# Create and train the RNN
rnn = SimpleRNN(input_size, hidden_size, output_size)
rnn.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate)

# Make predictions
predictions = rnn.predict(X_train)
print("Predictions:")
print(predictions)
