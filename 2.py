import csv
with open('trainingexamples.csv') as csvFile:
    data = [tuple(line) for line in csv.reader(csvFile)]
def Domain():
    D = []
    for i in range(len(data[0])):
        D.append(list(set([ele[i] for ele in data])))
    return D
D = Domain()
def consistant(h1, h2):
    for x, y in zip(h1, h2):
        if not (x == "?" or (x != "ɸ" and (x == y or y == "ɸ"))):
            return False
    return True
def candidate_elimination():
    G = {('?',)*(len(data[0]) - 1), }
    S = ['ɸ']*(len(data[0]) - 1)
    no = 0
    print("\n G[{0}]:".format(no), G)
    print("\n S[{0}]:".format(no), S)
    for item in data:
        no += 1
        inp, res = item[:-1], item[-1]
        if res in "Yy":
            i = 0 
            G = {g for g in G if consistant(g, inp)}
            for s, x in zip(S, inp):
                if not s == x:
                    S[i] = '?' if s != 'ɸ' else x
                i += 1
        else:
            S = S  # unaffected for this eg.
            Gprev = G.copy()
            for g in Gprev: 
                if g not in G: 
                    continue
                for i in range(len(g)):  
                    if g[i] == "?":  
                        for val in D[i]:
                            if inp[i] != val and val == S[i]:
                                g_new = g[:i] + (val,) + g[i+1:]
                                G.add(g_new)
                    else:
                        G.add(g)
                G.difference_update([h for h in G if
                                     any([consistant(h, g1) for g1 in G if h != g1])])
        print("\n G[{0}]:".format(no), G)
        print("\n S[{0}]:".format(no), S)
candidate_elimination()
