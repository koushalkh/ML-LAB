from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
x = len(twenty_train.target_names)
print("\n The number of categories:",x)
print("\n The %d Different Categories of 20Newsgroups\n" %x)
i=1
for cat in twenty_train.target_names:
 print("Category[%d]:" %i,cat)
 i=i+1
print("\n Length of train data is",len(twenty_train.data))
print("\n Length of file names is ",len(twenty_train.filenames))
#Considering only four Categories
categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train',categories=categories,shuffle=True)
twenty_test = fetch_20newsgroups(subset='test',categories=categories,shuffle=True)
print("Reduced length of train data",len(twenty_train.data))
print("length of test data",len(twenty_test.data))
print("Target Names",twenty_train.target_names)
#print("\n".join(twenty_train.data[0].split("\n")))
#print(twenty_train.target[0])
#Extracting features from text files
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
#Term Frequencies(tf): Divide the number of occurrences of each word in a document by the total number of words in the document
X_train_tf = count_vect.fit_transform(twenty_train.data)
X_train_tf.shape
print("tf train count",X_train_tf.shape)
#another refinement for tf is called tf–idf for “Term Frequency times Inverse Document Frequency”. 
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_tf)
X_train_tfidf.shape
print("tfidf train count",X_train_tfidf.shape)
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn import metrics
mod = MultinomialNB()
mod.fit(X_train_tfidf, twenty_train.target)
X_test_tf = count_vect.transform(twenty_test.data)
print("tf test count",X_test_tf.shape)
X_test_tfidf = tfidf_transformer.transform(X_test_tf)
print("tfidf test count",X_test_tfidf.shape)
predicted = mod.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(twenty_test.target, predicted))
print(classification_report(twenty_test.target,predicted,target_names=twenty_test.target_names))

print("confusion matrix is \n",metrics.confusion_matrix(twenty_test.target, predicted))
