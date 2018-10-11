import math
import pandas as pd
from collections import Counter
tennis_df = pd.DataFrame.from_csv('tennis.csv')
#Shuffle data OPTIONAL LINE if this line is used accuracy changes each time this code is executed!
tennis_df = tennis_df.sample(frac=1).reset_index(drop=True)
class Node:
    def __init__(self, data, attribute = None):
        self.decision_attribute = attribute
        self.child = {}
        self.data = data
        self.decision = None

def calculate_entropy(probs):
    return sum( [-prob*math.log(prob, 2) for prob in probs] )

def splitData(a_list,attribute,Class):
    return a_list[ a_list[attribute] == Class]

def entropy(a_list ,attribute = 'PlayTennis',  Gain = False):  
    cnt = Counter(a_list[attribute])   # Counter calculates the propotion of class
    num_instances = len(a_list[attribute])
    probs = [x / num_instances for x in cnt.values()]  # x means count of each attribute.
    if not Gain:
        return calculate_entropy(probs)
    print(cnt.items())
    gain = 0
    for Class , prob in zip(cnt.keys(),probs):
        gain += -prob *entropy(splitData(a_list,attribute,Class))
    return gain

def information_gain(data):
    Max_gain = -1
    Max_gain_Attribute = None
    for attribute in data.keys():  #Find max information gain
        if attribute == 'PlayTennis':
            continue
        gain = entropy(data)  + entropy(data,attribute,Gain= True)
        print("{} {}".format(gain,attribute))
        if gain > Max_gain:
            Max_gain = gain
            Max_gain_Attribute = attribute
    return Max_gain_Attribute  

def id3(root):
    global nodes 
    if len(root.data.keys()) == 1 or len(root.data) == 1:   #end of decision tree.
        cnt = Counter(root.data['PlayTennis'])
        root.decision = cnt.most_common(1)[0][0]  # Yes or No
        print("Decision=",root.decision)
        return
    Max_gain_Attribute = information_gain(root.data)
    root.decision_attribute = Max_gain_Attribute
    print("------------------------------")
    for attribute in set(root.data[Max_gain_Attribute]):
        #split data based on values in table
        childData = splitData(root.data , Max_gain_Attribute , attribute)
        root.child[attribute] = Node(childData.drop([Max_gain_Attribute],axis = 1))
        id3(root.child[attribute])
def predict(example,root):
    if root.decision != None:
        return root.decision
    try:   #if that attribute is not there in that part of tree(id3 trained on the split data)
        #eg: Temo(cold , hot , mild) but when divided into subtrees data is divided.
        #some branches have only cold and hot for eg.
        prediction = predict(example , root.child[example[root.decision_attribute]])
        return prediction
    except:
        return "No"

training_data = tennis_df.iloc[:-4] # all but last four instances
test_data  = tennis_df.iloc[-4:] # just the last four
root = Node(data = training_data)
id3(root)
print(test_data)
predictions = [predict(test ,root) for _,test in  test_data.iterrows() ]
correct = test_data['PlayTennis']
print("predictions are : {}".format(predictions))
print("actual:{}".format(test_data['PlayTennis']))
print("ACCURACY:{}".format(sum([1 for x,y in zip(predictions,correct) if x==y])/(len(predictions)) ))
###  OPTIONAL 
def display_tree(root):
    if root.decision != None:
        return  "{" + str(root.decision) + "},"
    tree ="{" +str(root.decision_attribute) + ":"
    for attribute , node  in root.child.items():
        tree += "{"+str(attribute)+":"+display_tree(node)+"},"
    return tree + "}"
print(display_tree(root))
