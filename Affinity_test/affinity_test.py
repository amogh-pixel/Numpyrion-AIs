import numpy as np

class Math_Neurons():
    def __init__(self,user_input,num_neuron):
        self.weights = []
        self.biases = []
        self.activations = []
        self.outputs = []
        self.user_input = user_input
        self.num_input = len(user_input)
        self.num_neuron = num_neuron
    
    def Leaky_ReLu(self,x):
        return np.where(x>0,x,x*0.01)
    def Leaky_ReLu_Derivative(self,x):
        dx = np.ones_like(x)
        dx[x<=0] = 0.01
        return dx
    def neuron(self,x,w,b):
        return np.dot(x,w)+b
    def clean_output(self, output):
        return output[output > 0]
    def test_neurons(self):
        while len(self.weights) < self.num_neuron:
                input  = self.user_input
                self.weight1 = np.random.randn(self.num_input,1)
                self.bias1 = np.zeros(1)
                self.weight2 = np.random.randn(self.num_input,1)
                self.bias2 = np.zeros(1)
                self.output1 = self.neuron(input,self.weight1,self.bias1)
                self.output2 = self.neuron(input,self.weight2,self.bias2)
                cleaned_output1 = self.clean_output(self.output1)
                cleaned_output2 = self.clean_output(self.output2)
                self.activation1 = self.Leaky_ReLu(cleaned_output1)
                self.activation2 = self.Leaky_ReLu(cleaned_output2)
                if np.sum(self.activation1)>np.sum(self.activation2):
                    self.weights.append(self.weight1)
                    self.biases.append(self.bias1)
                elif np.sum(self.activation2)>np.sum(self.activation1):
                    self.weights.append(self.weight2)
                    self.biases.append(self.bias2)
        return self.weights,self.biases

user_input = np.random.randn(5)
neuron_count = int(input("Enter number of neurons you want:"))
Neuron_finder = Math_Neurons(user_input,neuron_count)
weights,biases = Neuron_finder.test_neurons()
print(f"weights:\n{weights}\n\nbiases:\n{biases}")