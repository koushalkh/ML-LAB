import csv
with open('trainingexamples.csv') as csvFile:
    data = [line[:-1] for line in csv.reader(csvFile) if line[-1] == "Y"]
print("POSITIVE EXAMPLES ARE:{}".format(data))
S = ['ɸ']*len(data[0])   # Initializing.
print("output in each steps are:\n{}".format(S))
for example in data:
    i = 0
    for feature in example:
        S[i] = feature if S[i] == 'ɸ' or S[i] == feature else '?'
        i += 1
    print(S)
