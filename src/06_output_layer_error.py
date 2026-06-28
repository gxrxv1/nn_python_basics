import numpy as np
def sigmoid(x): 
    return 1/(1+np.exp(-x))
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
y=[[1]
   ,[0]]
a=[[0.8]
   ,[0.2]]
z2=np.array([[0.5],
             [0.7]])
nabla_cost= np.array(a) - np.array(y)
prime = sigmoid_prime(z2)
error = nabla_cost * prime
print(nabla_cost)
print(prime)
print(error)