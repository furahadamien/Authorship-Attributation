import math
import time
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from ordered_set import OrderedSet
from nltk.tokenize import sent_tokenize
from nltk.util import ngrams
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

sigma = 0.2
c = 0.01

def smoothing_fn(x, mu):
    if 0 <= x <= 1:
        return norm.pdf(x, mu, sigma) / (norm.cdf((1-mu)/sigma) - norm.cdf((0-mu)/sigma))
    else:
        return 0

def integrand(t, doc_x, term_j, mu, V):
    index = math.ceil(t*len(doc_x)) - 1 # adjust by -1 because arrays are 0-indexed 
    if term_j in doc_x[index]:
        return (doc_x[index][term_j] + c) / (1 + c*V) * smoothing_fn(t, mu)
    else:
        return c / (1 + c*V) * smoothing_fn(t, mu)
    
def D(P, Q):
    s = 0
    if len(P) != len(Q):
        return 0
    for i in range(0, len(P)):
        s += (P[i] - Q[i])*(P[i] - Q[i])

    return math.sqrt(s)

def K(P, Q):
    gamma = 1
    return math.exp(-1 * math.pow(D(P,Q),2) / gamma)

def my_kernel(A, B):
    matrix = [[0 for i in range(len(B))] for j in range(len(A)) ]
    for i in range(len(A)):
        for j in range(len(B)):
            matrix[i][j] = K(A[i], B[j])
    return matrix    

def load_corpus():
    X = []
    y = []
    for i in range(3):
        f = open("corpus/author_" + str(i) + ".txt")
        lines = f.readlines()[1:50] # skip the first line! includes author/title
        for l in lines:
            X.append(l)
            y.append(i)
        f.close()
    return X, y

def get_feature_matrix(corpus, vocabulary):
    X = []
    V = len(vocabulary)
    num_docs = 0

    for doc in corpus:
        print(str(num_docs+1) + "/" + str(len(corpus_train)))

        num_docs+=1
        x = []
        N = len(doc) - 3 + 1
        chrs = [ c for c in doc ]

        # loop over the N terms (trigrams) in the document and store binary value in hashmap  
        for term in ngrams(chrs, 3):
            vocab_dict = {}
            vocab_dict[str(term[0]) + str(term[1]) + str(term[2])] = 1
            x.append(vocab_dict)

        # loop over each component of the output feature vector
        lobow = []
        for i in range(V):
            val = 0
            for mu in np.arange(0.33, 1.01, 0.33):
                for t in np.arange(0, 1, 0.04):
                    val += integrand(t, x, vocabulary[i], mu, V)
            lobow.append(val)
        
        X.append(lobow)

    return X

start_time = time.time()
corpus, y = load_corpus()
corpus_train, corpus_test, y_train, y_test = train_test_split(corpus, y, test_size = 0.20)
print("number of lines in corpus_train: " + str(len(corpus_train)))
print("length of first line in corpus_train: " + str(len(corpus_train[0])))
print("first line in corpus_train: " + corpus_train[0])

vect = CountVectorizer(analyzer="char", ngram_range=(3,3), max_features=2500)
vect.fit_transform(corpus_train)

print("size of vocab: " + str(len(vect.get_feature_names())))

vocabulary = vect.get_feature_names()
X = get_feature_matrix(corpus_train, vocabulary)


clf = OneVsRestClassifier(SVC(kernel=my_kernel))
clf.fit(X, y_train)
print("accuracy: ")
print(clf.score(get_feature_matrix(corpus_test, vocabulary), y_test))
print("time passed to process " + str(len(corpus_train) + len(corpus_test)) + " train and test docs: ")
print(time.time() - start_time)
