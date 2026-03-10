import numpy as np
import mnist

mnist_images = mnist.train_images()
mnist_labels = mnist.train_labels()

class NumpyrionAI_trainer():
    def __init__(self, index,layers, mnist_images, mnist_labels):
        self.layers = layers
        self.weights = []
        self.biases = []
        self.image = mnist_images[index]
        self.user_input = self.image.reshape(784) / 255.0
        self.input_len = len(self.user_input)

        label = mnist_labels[index]
        self.target = np.zeros(10)
        self.target[label] = 1
        for i in range(len(layers)-1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2/layers[i])
            b = np.zeros(layers[i+1])
            self.weights.append(w)
            self.biases.append(b)
    
    def neuron(self,x,w,b):
        return np.dot(x,w)+b
    
    def Leaky_reLu(self,x):
        return np.where(x>0,x,x*0.01)
    
    def Leaky_reLu_Derivative(self,x):
        return np.where(x>0,1,0.01)
    
    def softmax(self,x):
        y = np.exp(x-np.max(x,axis=-1, keepdims= True))
        return y/np.sum(y,axis=-1, keepdims=True)
    
    def cross_entropy(self,target,prediction):
        prediction = np.clip(prediction, 1e-9, 1-1e-9)
        loss = -(np.sum(target*np.log(prediction)))
        return loss
    
    def forward_pass(self):
        self.z =[]
        self.a = []
        self.z.append(self.neuron(self.user_input,self.weights[0],self.biases[0]))
        self.a.append(self.Leaky_reLu(self.z[0]))
        for i in range(1,len(self.weights)):
            Z = self.neuron(self.a[i-1],self.weights[i],self.biases[i])
            self.z.append(Z)
            if i == len(self.weights)-1:
                A = self.softmax(self.z[i])
                self.a.append(A)
            else:
                A = self.Leaky_reLu(self.z[i])
                self.a.append(A)
        self.output = self.a[-1]
        return self.output
    def BackPropogation(self,mnist_image,mnist_labels,lr = 0.01,epochs=5):
        for epoch in range(epochs):
            for i in range(len(mnist_image)):
                self.user_input = mnist_image[i].reshape(784)/255.0
                label = mnist_labels[i]
                self.target = np.zeros(10)
                self.target[label] = 1
                y_pred = self.forward_pass().reshape(-1)
                error = y_pred-self.target
                loss = self.cross_entropy(self.target,y_pred)
                delta = error
                for l in reversed(range(len(self.weights))):
                    a_prev = self.user_input if l==0 else self.a[l-1]
                    dw = np.outer(a_prev,delta)
                    db = delta
                    self.weights[l] -= lr*dw
                    self.biases[l]-= lr*db

                    if l!=0:
                        delta = np.dot(delta,self.weights[l].T)*self.Leaky_reLu_Derivative(self.z[l-1])
            print(f"epoch {epoch} finished, loss:{loss:.6f}")
            
layers = [784,128,64,10]

trainer = NumpyrionAI_trainer(0, layers, mnist_images, mnist_labels)

trainer.BackPropogation(mnist_images, mnist_labels, epochs=5)