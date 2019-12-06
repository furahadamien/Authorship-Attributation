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

categories = ["candidate00001", "candidate00002", "candidate00003"
, "candidate00004", "candidate00005"]

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
print(test_data.target_names[test_data.target[0]])

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

# Support Vector Machine model
text_clf = Pipeline([('vect', CountVectorizer()),
('tfidf', TfidfTransformer()),
('clf', SGDClassifier(loss='hinge', penalty='l2',
alpha=1e-3, random_state=42,
max_iter=5, tol=None)),])
text_clf.fit(train_data.data, train_data.target)

#SVM evaluation
predicted = text_clf.predict(test_doc)
print('Support Vector Machine accuracy %r:' %np.mean(predicted == test_data.target) )
print('Support Vector Machin model confusion Matrix')
print(metrics.confusion_matrix(test_data.target, predicted))
print(train_data.target)
 


