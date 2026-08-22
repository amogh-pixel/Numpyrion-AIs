import numpy as np

class AI():
    def __init__(self,user,layer):
        self.weights = []
        self.biases = []
        self.user_input = user
        self.in_len = len(self.user_input)
        self.layers = layer
    
    def He_initialization(self,x):
        y = np.sqrt(2/x)
        return y

    def neuron(self,x,w,b):
        return np.dot(x,w)+b
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_derivative(self,x):
        return x* (1-x)
    
    def SwiGLU(self,x):
        y = x * self.sigmoid(x)
        return y
    
    def SwiGLU_derivative(self,x):
        a = self.sigmoid(x)
        y = x * self.sigmoid_derivative(a)
        z = a+y
        return z
    def weights_biases(self):
        weight1 = np.random.randn(self.in_len,self.layers[0])*self.He_initialization(self.in_len)
        bias1 = np.zeros(self.layers[0])
        self.weights.append(weight1)
        self.biases.append(bias1)
        for i in range(1,len(self.layers)):
            weight = np.random.randn(self.layers[i-1],self.layers[i])*self.He_initialization(self.layers[i-1])
            bias = np.zeros(self.layers[i])
            self.weights.append(weight)
            self.biases.append(bias)
        return self.weights,self.biases

    def forward_pass(self):
        self.outputs = []
        self.activations = []
        self.inputs = []
        self.inputs.append(self.user_input)
        output1 = self.neuron(self.user_input,self.weights[0],self.biases[0])
        act1 = self.SwiGLU(output1)
        self.outputs.append(output1)
        self.activations.append(act1)
        for i in range(1,len(self.layers)):
            layer_input = self.activations[i-1]
            output = self.neuron(self.activations[i-1],self.weights[i],self.biases[i])
            act = self.SwiGLU(output)
            self.outputs.append(output)
            self.activations.append(act)
            self.inputs.append(layer_input)
        return self.activations[-1]
    
    def backprop(self,target,lr = 0.01):
        error = self.activations[-1]-target
        for i in reversed(range(len(self.layers))):
            gradient = error*self.SwiGLU_derivative(self.outputs[i])
            error = gradient @ self.weights[i].T
            self.weights[i]-= lr*np.outer(self.inputs[i],gradient)
            self.biases[i]-=lr*gradient

# XOR training data
X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

Y = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])

# Create AI
ai = AI(X[0], [4, 4, 1])

# Initialize weights and biases
ai.weights_biases()

# Train
for epoch in range(10000):

    total_error = 0

    for x, target in zip(X, Y):

        # Give the AI a new input
        ai.user_input = x

        # Forward pass
        prediction = ai.forward_pass()

        # Calculate error
        error = np.mean((prediction - target) ** 2)
        total_error += error

        # Backpropagation
        ai.backprop(target, lr=0.01)

    if epoch % 1000 == 0:
        print("Epoch:", epoch, "Error:", total_error)

np.savez(
    "xor_model.npz",
    weight0=ai.weights[0],
    weight1=ai.weights[1],
    weight2=ai.weights[2],
    bias0=ai.biases[0],
    bias1=ai.biases[1],
    bias2=ai.biases[2]
)

# Test the trained AI
print("\nRESULTS")

for x in X:

    ai.user_input = x

    prediction = ai.forward_pass()

    print(x, "→", prediction)
