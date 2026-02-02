import numpy as np

# Creating Sigmoid function

def sigmoid(z):
    return 1/(1+np.exp(-z))

# Training data

x = np.array([1,2,3,4,5,6,7,8,9,10],dtype=float) # Score
y = np.array([0,0,0,0,1,1,1,1,1,1],dtype=float) # Pass/Fail

# Initializing brain

w = 0.0     # Weight
b = 0.0     # Bias
lr = 0.1   # Learning rate
n = len(x)  # Total size of our training data

for _ in range (10000):
    score = w*x + b # calculating the score
    y_pred = sigmoid(score) # converting the score into probability
    error = y_pred - y  # finding the error amount
    dw = (1/n) * np.sum(error*x)
    db = (1/n) * np.sum(error)
    w = w - lr * dw
    b = b - lr * db

x_test = 7
prob = sigmoid(w * x_test + b)
if prob>0.5:
    print("Pass")
else:
    print("fail")

