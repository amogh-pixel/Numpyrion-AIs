import numpy as np

class NumpyrionAI_trainer():
    def __init__(self,layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        self._init_params()
    def _init_params(self):
        for i in range(len(self.layers)-1):
            W = np.random.randn(self.layers[i],self.layers[i+1]).astype(np.float32)*0.1
            B = np.zeros((1,self.layers[i+1]),dtype=np.float32)

            self.weights.append(W)
            self.biases.append(B)
    def neuron(self,x,w,b):
        return np.dot(x,w)+b
    def ReLu(self,x):
        return np.maximum(0,x)
    def ReLu_derivative(self,x):
        return (x>0).astype(np.float32)   
    def lossMSE(self,x):
        return np.mean(np.square(x))
    def softmax(self,x):
        y = np.exp(x-np.max(x,axis = -1,keepdims=True))
        return y/np.sum(y,axis =-1,keepdims=True)
    def cross_entropy(self,output,target):
        esp = 1e-9
        return -np.mean(np.sum(target*np.log(output+esp),axis =1))
    def ForwardPass(self,x):
        self.z =[]
        self.a = [x]
        for i in range(len(self.weights)):
            self.z_current = self.neuron(self.a[i],self.weights[i],self.biases[i])
            if i == len(self.weights)-1:
                self.a_current = self.softmax(self.z_current)
            else:
                self.a_current = self.ReLu(self.z_current)
            self.z.append(self.z_current)
            self.a.append(self.a_current)

        return self.a[-1]
    def BackPropogation(self,target,lr = 0.01):
        error = self.a[-1]-target
        m = target.shape[0]
        self.loss = self.cross_entropy(self.a[-1],target)
        for i in reversed(range(len(self.weights))):
            if i != len(self.weights)-1:
                error = error*self.ReLu_derivative(self.z[i])
            dw = self.a[i].T @ error/m
            db = np.sum(error,axis=0,keepdims = True)/m
            error = error @ self.weights[i].T
            self.weights[i]-=lr*dw
            self.biases[i]-=lr*db

# 3-class classification
nn = NumpyrionAI_trainer([2, 16, 3])

X = np.random.randn(100, 2)

# one-hot labels
y = np.zeros((100, 3))
y[np.arange(100), np.random.randint(0, 3, 100)] = 1

for epoch in range(1000):
    nn.ForwardPass(X)
    nn.BackPropogation(y)

    if epoch % 50 == 0:
        print("loss:", nn.loss)

