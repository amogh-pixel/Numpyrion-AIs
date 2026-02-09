import numpy as np

class AIFunctions():
    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    @staticmethod
    def sigmoid_derivative(z):
        return z*(1-z)
    @staticmethod
    def neuron(y,w,b):
        return np.dot(y,w)+b
    @staticmethod
    def softmax(a):
        b = np.exp(a-np.max(a,axis=-1,keepdims=True))
        return b/np.sum(b,axis=-1,keepdims=True)
    @staticmethod
    def self_Attention(c):
        d = np.dot(c,c.T)
        d1 = AIFunctions.softmax(d)
        x1 = np.dot(d1,c)
        return x1
    @staticmethod
    def relu(x):
        return np.maximum(0,x)
    @staticmethod
    def relu_derivative(x):
        return (x>0).astype(float) 
    @staticmethod
    def MSE(error):
        return np.mean(np.square(error))
    