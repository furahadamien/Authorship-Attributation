from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from nltk import FreqDist
import numpy as np
import nltk
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from nltk import classify 
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn import model_selection, naive_bayes, svm
from sklearn.linear_model import SGDClassifier
import string
import math

from sklearn.model_selection import train_test_split

tfidf_vect= TfidfVectorizer(  use_idf=True, smooth_idf=True, sublinear_tf=False)


categories = ["candidate00001", "candidate00002", "candidate00003",
"candidate00004", "candidate00005", "candidate00006", "candidate00007",
"candidate00008", "candidate00009", "candidate00010", "candidate00011", 
"candidate00012", "candidate00013", "candidate00014", "candidate00015", 
"candidate00016", "candidate00017", "candidate00018", "candidate00019", 
"candidate00020", "candidate00021", "candidate00022", "candidate00023", 
"candidate00024", "candidate00025", "candidate00026", "candidate00027", 
"candidate00028", "candidate00029", "candidate00030"]

train_directory = '../Copora - furaha/Training1'
train_data = load_files(train_directory)

test_directory = '../Copora - furaha/Testing1'
test_data = load_files(test_directory)

test_doc = test_data.data
#print(test_doc)
#for f in test_doc:
    #print(f)
    #f.decode('utf-8')
test = "curtesy of Damien"


print(type(test_doc))
print(test_data.target_names)
#print(test_data.labels)

for x in range (0, len(test_doc)):
    token = nltk.word_tokenize(test_doc[x].decode('utf-8'))
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in token]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    #print(words[:100])
    #print(words[:100])
    #print(token.punctuation)
    #print(nltk.pos_tag(token))
count_vect = CountVectorizer()

gamma = 0.1

def Dd(P, Q):
    s = 0
    if len(P) != len(Q):
        return 0
    
    for i in range(1, len(P) + 1):
        s += math.acos(math.sqrt(P[i]) * math.sqrt(Q[i]))
    return s

def D(P, Q):
    s = 0
    if len(P) != len(Q):
        return 0
    for i in range(0, len(P)):
        s += (P[i] - Q[i])*(P[i] - Q[i])

    return math.sqrt(s)

def K(P, Q):
    return math.exp(-1 * math.pow(D(P,Q),2) / gamma)

# for LOWBOW: each row of P contains the unweighted sum over BOLH
def my_kernel(A, B):
    matrix = [[0 for i in range(len(B))] for j in range(len(A)) ]
    for i in range(len(A)):
        for j in range(len(B)):
            matrix[i][j] = K(A[i], B[j])
    return matrix


h = .02  # step size in the mesh

#stop_words = stopwords.words('english')
#stop_words = set(stopwords.words('english'))
#print(string.punctuation)

X_train_counts = count_vect.fit_transform(train_data.data)
print(X_train_counts.shape)
print(count_vect.vocabulary_.get(u'algorithm'))
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


df= pd.DataFrame({'text':test_doc, 'class': test_data.target})

X = tfidf_vect.fit_transform(df['text'].values)
y = df['class'].values

from sklearn.decomposition.truncated_svd import TruncatedSVD 
pca = TruncatedSVD(n_components=2)                                
X_reduced_train = pca.fit_transform(X)
a_train, a_test, b_train, b_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.ensemble import RandomForestClassifier 
classifier=RandomForestClassifier(n_estimators=10)                  
classifier.fit(a_train.toarray(), b_train)                            
prediction = classifier.predict(a_test.toarray()) 

clf = svm.SVC(kernel=my_kernel)
# Support Vector Machine model
#text_clf = Pipeline([('vect', CountVectorizer()),
#(#'tfidf', TfidfTransformer()),
#('clf', clf),])

clf.fit(a_train.toarray(), b_train)

#SVM evaluation
#predicted = clf.predict(test_doc)
print('Support Vector Machine accuracy %r:' %np.mean(prediction == b_train) )
print('Support Vector Machin model confusion Matrix')
print(metrics.confusion_matrix(b_train, prediction))
