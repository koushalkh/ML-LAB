#Program Written by KOUSHAL K H.
import numpy as np
def forward_propagation(X,W1,b1,W2,b2):
    Z1 = np.dot(W1, X)+b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1)+b2
    A2 = 1/(1+np.exp(-Z2)) #Sigmoid
    return A1,A2

def nn_model(X, Y, n_h, num_iterations=250,lr = 1.2):
    n_x, n_y, m = X.shape[0], Y.shape[0], Y.shape[1]
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    for i in range(0, num_iterations):
        #Forward Pass
        A1, A2 = forward_propagation(X, W1, b1, W2, b2)
        logprobs = np.multiply(np.log(A2), Y)+np.multiply(np.log(1-A2), (1-Y)) #LOSS Y*log(A2) + (1-Y)*log(1-A2)
        cost = (-1/m)*np.sum(logprobs)
        cost = np.squeeze(cost) #To make sure dimentions are proper
        #BackwardProp
        dZ2 = A2-Y
        dW2 = (1/m)*np.dot(dZ2, A1.T)
        db2 = (1/m)*np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.multiply(np.dot(W2.T, dZ2), (1 - np.power(A1, 2)))
        dW1 = (1/m)*np.dot(dZ1, X.T)
        db1 = (1/m)*np.sum(dZ1, axis=1, keepdims=True)
        #Update parameters
        W1 = W1 - lr*dW1
        b1 = b1 - lr*db1
        W2 = W2 - lr*dW2
        b2 = b2 - lr*db2
        print("Cost after iteration %i: %f" % (i, cost))
    return W1, b1, W2, b2
    
def predict(X,W1, b1, W2, b2):
    A2 = forward_propagation(X, W1, b1, W2, b2)
    predictions = A2[1] > 0.5
    return predictions

#Generating input for NN can replace next 4 lines with any input of your own
X = np.random.randn(2, 400)*15
Y = np.array([(1 if np.sum(a) > 0 else 0) for a in np.rollaxis(X, 1)])
Y[:70] = 1  # Adding Noice. Remove this line or decrease 70 for more accuracy
Y = Y.reshape(1, 400)
n_x, n_h, n_y = 2, 4, 1
W1, b1, W2, b2 = nn_model(X, Y, n_h)  #Train the model
predictions = predict(X,W1, b1, W2, b2)
print("ACUURACY={}%".format(np.sum((predictions == Y),axis = 1)/float(Y.size)*100))
"""
NOTE:
If no of layers is more than 2:  
    copy and paste the lines of 2nd layer and make it 3
To increase the accuracy:
    increase num_iterations
If any other input data is used:
    Accuracy may differ.
"""