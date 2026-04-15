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
            
user = np.array([0.2,0.5,0.7,0.1,0.9],dtype=np.float32)
layers = [6,5,4,1]
target = np.array([0.93],dtype=np.float32)

nn = AI(user, layers)
nn.weights_biases()

for epoch in range(400):
    output = nn.forward_pass()
    loss = nn.backprop(target)

    print(f"Epoch: {epoch} | Output: {output} ")
