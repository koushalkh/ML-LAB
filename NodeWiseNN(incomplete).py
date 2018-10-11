import random
from math import exp
class Node:
    def __init__(self,input_shape = None ):
        self.Weights = [random.uniform(-0.5,0.5) for _ in range(input_shape)] #np.random(input_shape)
        self.c = 0  # bias (mx + "c")
    def activate(self,Input):
        # weights[i]*input[i] + b like (mx + c)
        self.Z = []
        self.activation = []
        for inp in Input:
            self.Z.append(sum([i*j + self.c for i,j in zip(self.Weights, inp) ]))
            self.activation.append( 1.0 / (1.0 + exp(-self.Z[-1])))  # Applying Sigmoid activation function.
        return self.activation
class Layer:
    def __init__(self, noOfNodes = 1 , input_shape = None):
        self.noOfNodes = noOfNodes
        nodes = []
        for _ in range(noOfNodes):
            nodes.append(Node(input_shape= input_shape))
        self.Nodes = nodes

    def output(self , Input):
        temp = []
        for node in self.Nodes:
            temp.append(node.activate(Input))
        self.outputs = [[temp[j][i] for j in range(len(temp))] for i in range(len(temp[0]))] #Transpose of Matrix
        return self.outputs

def initialize_Network(input_size , nHiddenLayerNodes , outputSize = 1):
    layers = []
    prevLayer = input_size
    for nNodes in nHiddenLayerNodes: #Hidden layers
        layers.append( Layer(nNodes , prevLayer))
        prevLayer = nNodes
    layers.append( Layer(outputSize , prevLayer)) #OutputLayer
    return layers
def forwardPropogation(Input ,layers):
    activation = Input  #input of 1st layer is input itself 
    for layer in layers:
        activation = layer.output(activation)  # input of ith layer is activation of (i-1)th layer
    return [a[0] for a in activation]  #activation of final layer is the output of the NN
def loss(ouput , ExpectedOutput):
    n = len(ExpectedOutput)
    return (1/(2*n))*sum([(ExpectedOutput[i]-output[i])**2 for i in range(n)])


"""
@TODO : average over loss of every example(sum of activation vector) and find gradient and subtract it with respective weights.
"""
def backPropogation(layers , loss , lr = 0.01):
    
    for layer in reversed(layers):
        
        for node in layer:
            
            



"""
Test :
  input = 3
  n Hidden layers = 4
  hidden layer nodes: [3 , 3 , 2 , 2]
  output shape = 1
"""
Input = [[random.randint(0,10) for _ in range(3)] for _ in range(1000)]
correctOutput = [random.randint(0,1) for _ in range(1000)]
# Just  2 lines of code for any type of network forward pass!!
layers = initialize_Network(3 , [3 , 3, 2] , 1)
output =forwardPropogation(Input,layers)
print(len(output),output)
#print(output,correctOutput)
print(loss(output , correctOutput))