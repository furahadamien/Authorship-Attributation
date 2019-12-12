import math
import time
import numpy as np
from os.path import exists
from scipy.integrate import quad
from scipy.stats import norm
from nltk.tokenize import sent_tokenize
from nltk.util import ngrams
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix

sigma = 0.2     # smaller sigma == more sequential information
c = 0.01        # small value used to smooth the lobow

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

def load_corpus(dir_path, L):
    X = []
    y = []
    for i in range(L): # there are L documents in folder L to read in; file names are indexed by i
        f = open(dir_path + "/" + str(L) + "/author_" + str(i) + ".txt")
        lines = f.readlines()
        for ln in lines:
            X.append(ln)
            y.append(i)
        f.close()
    return X, y

def get_feature_matrix(corpus, vocabulary, savename):
    X = []
    V = len(vocabulary)
    counter = 1
    
    for doc in corpus:
        print(str(counter) + "/" + str(len(corpus)))
        counter+=1
        
        x = []
        N = len(doc) - 3 + 1
        chrs = [ c for c in doc ]

        # loop over the N terms (trigrams) in the document and store binary value in hashmap  
        for term in ngrams(chrs, 3):
            vocab_dict = {}
            vocab_dict[str(term[0]) + str(term[1]) + str(term[2])] = 1
            x.append(vocab_dict)

        # loop over each component of the output feature vector and build by summing over kernel positions
        lobow = []
        for i in range(V):
            val = 0
            for mu in np.arange(0.33, 1.01, 0.33):
                for t in np.arange(0, 1, 0.1):
                    val += integrand(t, x, vocabulary[i], mu, V)
            lobow.append(val)
        X.append(lobow)
    np.savetxt(savename, X, fmt='%s')
    return np.array(X)

def run_test(path, L):
    start_time = time.time()

    corpusA, yA = load_corpus("corpus_gb", L)
    corpusB, yB = load_corpus(path, L)
    
    corpus_trainA, corpus_testA, y_trainA, y_testA = train_test_split(corpusA, yA, test_size=0.20)
    corpus_trainB, corpus_testB, y_trainB, y_testB = train_test_split(corpusB, yB, test_size=0.20)
    
    print("lenA train: " + str(len(corpus_trainA)))
    print("lenB train: " + str(len(corpus_trainB)))
    print("lenA test: " + str(len(corpus_testA)))
    print("lenB test: " + str(len(corpus_testB)))
    
    corpus_train = corpus_trainA + corpus_trainB
    corpus_test = corpus_testA + corpus_testB
    y_train = y_trainA + y_trainB
    y_test = y_testA + y_testB 

    print("number of lines in full corpus: " + str(len(corpus)))
    print("train: " + str(len(corpus_train)))
    print("length of first line in corpus_train: " + str(len(corpus_train[0])))
    print("first line in corpus_train: " + corpus_train[0])

    vect = CountVectorizer(analyzer="char", ngram_range=(3,3), max_features=2500)
    vect.fit_transform(corpus_train)

    print("size of vocab: " + str(len(vect.get_feature_names())))
    vocabulary = vect.get_feature_names()
    
    savename = "X_train_" + path + "_" + str(L) 
    X = get_feature_matrix(corpus_train, vocabulary, savename)

    print("training the model")
    clf = OneVsRestClassifier(SVC(kernel=my_kernel))
    clf.fit(X, y_train)

    print("testing the model")
    savename = "X_test_" + path + "_" + str(L) 
    X_test = get_feature_matrix(corpus_test, vocabulary, savename)
    y_pred = clf.predict(X_test)

    f = open("results/" + path + "-" + str(L) + "-results (" + time.strftime("%H_%M_%S") + ")", "w")
    f.write(classification_report(y_test, y_pred, zero_division=0))
    f.write("\n")
    f.write(np.array2string(multilabel_confusion_matrix(y_test, y_pred), separator=', '))
    f.write("\n\ntime passed: ")
    f.write(str(time.time() - start_time))
    f.write("\n# authors: " + str(L))
    f.write("\n# train docs: " + str(len(corpus_train)))
    f.write("\n# test docs: " + str(len(corpus_test)))
    f.write("\nvocab size: " + str(len(vocabulary)))
    f.close()


if __name__ == "__main__":
    
    pair2 = [ "corpus_wp", "corpus_blog", "corpus_hp" ]
    
    for L in [ 2, 3 ]:
        for corpus in pair2:
            try:
                run_test(corpus, L)
            except Exception as e:
                print(e)
                print("Something went wrong! Failure for test: " + corpus + " " + str(L))