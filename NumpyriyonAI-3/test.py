import numpy as np


class AI:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def SwiGLU(self, x):
        return x * self.sigmoid(x)

    def neuron(self, x, w, b):
        return np.dot(x, w) + b

    def forward_pass(self, user_input):

        x = user_input

        for i in range(len(self.weights)):

            x = self.neuron(
                x,
                self.weights[i],
                self.biases[i]
            )

            x = self.SwiGLU(x)

        return x


# Load trained model
model = np.load("xor_model.npz")

weights = [
    model["weight0"],
    model["weight1"],
    model["weight2"]
]

biases = [
    model["bias0"],
    model["bias1"],
    model["bias2"]
]


# Create AI using ONLY the saved parameters
ai = AI(weights, biases)


# Test data
X_test = np.array([
    [0.2, 0.0],
    [0.8, 0.1],
    [0.2, 0.5],
    [0.8, 0.8],
    [0.1, 0.2],
    [0.9, 1.0]
])

for x in X_test:
    prediction = ai.forward_pass(x)
    print(x, "→", prediction)
