from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from nltk import FreqDist
import numpy as np
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

categories = ["candidate00001", "candidate00002", "candidate00003",
"candidate00004", "candidate00005", "candidate00006", "candidate00007",
"candidate00008", "candidate00009", "candidate00010", "candidate00011", 
"candidate00012", "candidate00013", "candidate00014", "candidate00015", 
"candidate00016", "candidate00017", "candidate00018", "candidate00019", 
"candidate00020", "candidate00021", "candidate00022", "candidate00023", 
"candidate00024", "candidate00025", "candidate00026", "candidate00027", 
"candidate00028", "candidate00029", "candidate00030", "candidate00031", 
"candidate00032", "candidate00033", "candidate00034", "candidate00035", 
"candidate00036", "candidate00037", "candidate00038", "candidate00039", 
"candidate00040", "candidate00041", "candidate00042", "candidate00043",
"candidate00044", "candidate00045", "candidate00046", "candidate00047", 
"candidate00048", "candidate00049", "candidate00050", "candidate00051",
"candidate00052", "candidate00053", "candidate00054", "candidate00055", 
"candidate00056", "candidate00057", "candidate00058", "candidate00059",
"candidate00060", "candidate00061", "candidate00062", "candidate00063",
"candidate00064", "candidate00065", "candidate00066", "candidate00067", 
"candidate00068", "candidate00069", "candidate00070", "candidate00071",
"candidate00072"]

train_directory = '../Copora - furaha/Training'
train_data = load_files(train_directory)

test_directory = '../Copora - furaha/Testing'
test_data = load_files(test_directory)

test_doc = test_data.data

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_data.data)

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



#for x in range (10, 73):
#    print('"candidate000' + str(x)+ '", ', end="")

