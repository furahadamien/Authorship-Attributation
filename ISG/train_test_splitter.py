from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from os import stat
from os import mkdir 

k_path = "Code/src/resources/known/"
uk_path = "Code/src/resources/unknown/"

def load_corpus(dir_path, L, start=0):
    X = []
    y = []
    # wnl = WordNetLemmatizer()
    for i in range(L): # there are L documents in folder L to read in; file names are indexed by i
        f = open(dir_path + "/" + str(L) + "/author_" + str(i) + ".txt")
        lines = f.readlines()
        for ln in lines:
            """words = word_tokenize(ln)
            lemmatized = ""
            for w in words:
                lemmatized += wnl.lemmatize(w) + " "
            X.append(lemmatized)"""
            X.append(ln)
            y.append(start + i)
        f.close()
    return X, y
    
def save_train_test(path, L):
    
    """
    corpusA, yA = load_corpus("corpus_gb", L, 0)
    corpusB, yB = load_corpus(path, L, L)
    
    X_trainA, X_testA, y_trainA, y_testA = train_test_split(corpusA, yA, test_size=0.20)
    X_trainB, X_testB, y_trainB, y_testB = train_test_split(corpusB, yB, test_size=0.20)
    
    y_trainA, X_trainA = (list(t) for t in zip(*sorted(zip(y_trainA, X_trainA))))
    y_trainB, X_trainB = (list(t) for t in zip(*sorted(zip(y_trainB, X_trainB))))
    
    X_train = X_trainA + X_trainB
    X_test = X_testA + X_testB 
    y_train = y_trainA + y_trainB 
    y_test = y_testA + y_testB
    """
    
    corpus, y = load_corpus(path, L)
    X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size=0.20)
    y_train, X_train = (list(t) for t in zip(*sorted(zip(y_train, X_train))))
 
    is_open = ""
    idn = 0
    f = None
    for i in range(len(y_train)):
        author = y_train[i]
        if is_open != author:
            if f is not None: f.close()
            
            full_dir = k_path + path + "/" + str(L) + "/"
            try:
                stat(full_dir)
            except:
                mkdir(full_dir)
            
            f = open(full_dir + "author_" + str(author) + ".txt", "w+")
            is_open = author

        f.write(X_train[i])
    f.close()
    
    counter = 0
    
    full_dir = uk_path + path + "/" + str(L) + "/"
    try:
        stat(full_dir)
    except:
        mkdir(full_dir)    
    g = open(full_dir + "labels.txt", "w+")
    for ln in X_test:
        f = open(full_dir + "test_" + str(counter) + ".txt", "w+")
        g.write("author_" + str(y_test[counter]) + "\n")
        counter += 1
        f.write(ln)
        f.close()
    g.close()
   
if __name__ == "__main__":
    corpora = [ "corpus_gb", "corpus_hp", "corpus_wp", "corpus_blog" ]
    for L in [ 3, 5, 10 ]:
        for corpus in corpora:
            save_train_test(corpus, L)
    
    """
    for L in [3]: # [ 3, 5, 10 ]:
        save_train_test("corpus_wp", L)
    
    corpora = [ "corpus_hp", "corpus_wp", "corpus_blog" ]
    for corpus in corpora:
        for L in [ 3, 10, 50 ]:
            save_train_test(corpus, L)
    """