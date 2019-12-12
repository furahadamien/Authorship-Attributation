from scipy.stats import norm
from nltk.util import ngrams
from ordered_set import OrderedSet
docs = [ "My, my, I was forgetting all about the children and the mysterious fern seed." ]

def smoothing_fn(x, mu, sigma):
    print(norm.pdf(x, mu, sigma))
    print(norm.cdf(x))

    if 0 <= x <= 1:
        return gauss.pdf(x) / (norm.cdf((1-mu)/sigma) - norm.cdf((0-mu)/sigma))
    else:
        return 0

W = []
V = []
T = []
D = []
vocabulary = OrderedSet([])

for d in docs:
    N = len(d) - 3 + 1 # for n = 3 (character level trigrams)
    chrs = [ c for c in d]

    features = []
    indices = []
    bag_of_hist = {}
    for n in ngrams(chrs, 3):
        features.append(str(n[0]) + str(n[1]) + str(n[2]))
        # add(...) returns the index in the ordered set; append this to indices vector V
        index = (vocabulary.add(str(n[0]) + str(n[1]) + str(n[2])))
        indices.append(index)
        
        if index not in bag_of_hist: bag_of_hist[index] = 0
        bag_of_hist[index] += 1
        
    W.append(features)
    V.append(indices)
    D.append(bag_of_hist)

    t = []
    next_t = 0
    for i in range(N):
        t.append(next_t)
        next_t += 1/N

    T.append(t)

    smoothing_fn(0.5, 1, 0.1)

    """BOLH = []
    for i in range(len(T)):
        BOLH.append(D[V[i]] * smoothing_fn(T[i]))

    for local_hist in BOLH:
        print(local_hist)
    """
   
print(vocabulary)
print("W")
print(W)
print("V")
print(V)
print("T")
print(T)
