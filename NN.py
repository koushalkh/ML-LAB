import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))

def layer_sizes(X, Y):
    n_x = X.shape[0]  # size of input layer
    n_h = 4 #size of hidden layers.
    n_y = Y.shape[0]  # size of output layer
    return (n_x, n_h, n_y)

def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    parameters = {"W1": W1, "b1": b1,"W2": W2,"b2": b2}
    return parameters

def forward_propagation(X, parameters):
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    Z1 = np.dot(W1, X)+b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1)+b2
    A2 = sigmoid(Z2)
    cache = {"Z1": Z1,"A1": A1,"Z2": Z2,"A2": A2}
    return A2, cache

def compute_cost(A2, Y, parameters):
    m = Y.shape[1]  # number of example
    logprobs = np.multiply(np.log(A2), Y)+np.multiply(np.log(1-A2), (1-Y))
    cost = (-1/m)*np.sum(logprobs)
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect.
    return cost

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1, W2 = parameters["W1"], parameters["W2"]
    # Retrieve also A1 and A2 from dictionary "cache".
    A1, A2 = cache["A1"], cache["A2"]
    dZ2 = A2-Y
    dW2 = (1/m)*np.dot(dZ2, A1.T)
    db2 = (1/m)*np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), (1 - np.power(A1, 2)))
    dW1 = (1/m)*np.dot(dZ1, X.T)
    db1 = (1/m)*np.sum(dZ1, axis=1, keepdims=True)
    grads = {"dW1": dW1,"db1": db1,"dW2": dW2, "db2": db2}
    return grads

def update_parameters(parameters, grads, learning_rate=1.2):
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    # Retrieve each gradient from the dictionary "grads"
    dW1, db1 = grads["dW1"], grads["db1"]
    dW2, db2 = grads["dW2"], grads["db2"]
    # Update rule for each parameter
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    parameters = {"W1": W1,"b1": b1,"W2": W2,"b2": b2}
    return parameters

def nn_model(X, Y, n_h, num_iterations=10000, print_cost=True):
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        # Gradient descent parameter update
        parameters = update_parameters(parameters, grads)
        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    return parameters
def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = A2 > 0.5
    return predictions
#Generating input for NN
X = np.random.randn(2,400)*15
Y = np.array([(1 if np.sum(a) > 0 else 0) for a in np.rollaxis(X, 1)])
Y[:70] = 1 #Adding Noice. Remove this line or decrease 70 for more accuracy
Y = Y.reshape(1,400)
m = X.shape[1]
n_x, n_h, n_y = 2 , 4 , 1
parameters = nn_model(X, Y, n_h, num_iterations=5000)
predictions = predict(parameters, X)
print('Accuracy: %d' % float((np.dot(Y, predictions.T) +
             np.dot(1-Y, 1-predictions.T))/float(Y.size)*100) + '%')
