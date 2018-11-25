import csv
import math

def mean(numbers):
	return sum(numbers)/float(len(numbers))
	
def stdev(numbers):
	avg = mean(numbers)		
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

#Summarization yields the mean and standard deviation attribute wise
def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

#using Gaussian NB Probability density formula	
def calcProb(summary, item):
	prob = 1
	for i in range(len(summary)):
		x = item[i]
		mean, stdev = summary[i]		
		exponent = math.exp(-math.pow(x-mean,2) / (2*math.pow(stdev,2)))
		final = exponent / (math.sqrt(2*math.pi) * stdev)
		prob *= final
	return prob
	
with open('ConceptLearning.csv') as csvFile:
	data = [line for line in csv.reader(csvFile)]
for i in range(len(data)):
	data[i] = [float(x) for x in data[i]]

split = int(0.90 * len(data)) #Split Ratio
train = []
test = []
train = data[:split] #Training dataset
test = data[split:] #Testing dataset

print("{} input rows is split into {} training and {} testing datasets".format(len(data), len(train), len(test)))
print("\nThe values assumed for the concept learning attributes are\n")
print("OUTLOOK=> Sunny=1 Overcast=2 Rain=3\nTEMPERATURE=> Hot=1 Mild=2 Cool=3\nHUMIDITY=> High=1 Normal=2\nWIND=> Weak=1 Strong=2")
print("TARGET CONCEPT:PLAY TENNIS=> Yes=10 No=5")
print("\nThe Training set are:")
for x in train:
	print(x)
print("\nThe Test data set are:")
for x in test:
	print(x)

yes = [] #'yes' corresponds to 'playTennis=Yes' inputs
no = [] #'no' corresponds to 'playTennis=No' inputs
for i in range(len(train)):
	if data[i][-1] == 5.0:
		no.append(data[i])
	else:
		yes.append(data[i])

#Summarizing both 'yes' and 'no' individually
yes = summarize(yes)
no = summarize(no)

predictions = [] #contains predicted values, yes(10.0) or no(10.0)
for item in test:
	yesProb = calcProb(yes, item) #evaluating probability of test data being 'yes'
	noProb = calcProb(no, item) #evaluating probability of test data being 'no'
	predictions.append(10.0 if(yesProb > noProb) else 5.0) #choosing higher probability class(yes or no)

correct = 0
for i in range(len(test)):
	if(test[i][-1] == predictions[i]):
		correct += 1
		
print("\nActual values are:")
for i in range(len(test)):
	print(test[i][-1], end=" ")
print("\nPredicted values are:")
for i in range(len(predictions)):
	print(predictions[i], end=" ")
print("\nAccuracy is {}%".format(float(correct/len(test) * 100)))