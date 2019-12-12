from scipy.stats import norm
from nltk.util import ngrams
from ordered_set import OrderedSet
docs = [ "My, my, I was forgetting all about the children and the mysterious fern seed." ]

def smoothing_fn(x, mu, sigma):
    print(norm.pdf(x, mu, sigma))
    print(norm.cdf(x))

    if 0 <= x <= 1:
        return gauss.pdf(x) / (norm.cdf((1-mu)/sigma) - norm.cdf((0-mu)/sigma))
    else
        return 0

W = []
V = []
T = []
vocabulary = OrderedSet([]) 
for d in docs:
    N = len(d) - 3 + 1 # for n = 3 (character level trigrams)
    chrs = [ c for c in d]

    features = []
    indices = []
    for n in ngrams(chrs, 3):
        features.append(str(n[0]) + str(n[1]) + str(n[2]))
        indices.append(vocabulary.add(str(n[0]) + str(n[1]) + str(n[2])))
        
    W.append(features)
    V.append(indices)

    t = []
    next_t = 0
    for i in range(N):
        t.append(next_t)
        next_t += 1/N

    T.append(t)


    


print(vocabulary)
print("W")
print(W)
print("V")
print(V)
print("T")
print(T)
