from AI_functions import AIFunctions
from Custom_Embedder import Embed_tool
import numpy as np

user = input("Enter input:")
target = input("Enter target:")
cleaned_user = Embed_tool.clean(user)
cleaned_target = Embed_tool.clean(target)
embed_user = Embed_tool.encode(cleaned_user)
embed_target = Embed_tool.encode(cleaned_target)
x = np.array(embed_user).reshape(1,-1)
t = np.array(embed_target).reshape(1,-1)
lr = 0.01
w = []
b = []
num_neurons = 2048
num_layers = 30
out_size = t.shape[1]
in_len = x.shape[1]
x1 = AIFunctions.self_Attention(x)
w.append(np.random.randn(in_len,num_neurons)*0.1)
b.append(np.random.randn(1,num_neurons))
for i in range(1,num_layers-1):
    w.append(np.random.randn(num_neurons,num_neurons)*0.1)
    b.append(np.random.randn(1,num_neurons))
w.append(np.random.randn(num_neurons,out_size)*0.1)
b.append(np.random.randn(1,out_size))

z =[AIFunctions.neuron(x1,w[0],b[0])]
a = [AIFunctions.sigmoid(z[0])]
for i in range(10000):
    for i in range(1,num_layers-1):
        z.append(AIFunctions.neuron(a[i-1],w[i],b[i]))
        a.append(AIFunctions.sigmoid(z[i]))
    z.append(AIFunctions.neuron(a[-1],w[-1],b[-1]))
    a.append(AIFunctions.sigmoid(z[-1]))
    error = a[-1]-t
    loss =AIFunctions.MSE(error)
    for i in reversed(range(len(w))):
        last_act = a[i-1] if i>0 else x1
        dw = np.dot(last_act.T,error)
        db = np.sum(error,axis=0,keepdims=True)
        if i>0:
            error = np.dot(error,w[i].T)*AIFunctions.sigmoid_derivative(a[i-1])
        w[i]-=lr*dw
        b[i]-=lr*db
print(a[-1])
print(w)
print(b)
print(f"you asked: {cleaned_user}")
answer = Embed_tool.decode(a[-1].flatten())

print(f"AI responded: {answer}")
