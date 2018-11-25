from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np

categories=['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
news_train=fetch_20newsgroups(subset='train',categories=categories,shuffle='true')
news_test=fetch_20newsgroups(subset='test',categories=categories,shuffle='true')
print("Target Names",news_train.target_names)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

text_clf=Pipeline([('vect',TfidfVectorizer()),('clf',MultinomialNB())])
text_clf.fit(news_train.data,news_train.target)
predicted=text_clf.predict(news_test.data)

from sklearn import metrics
from sklearn.metrics import accuracy_score
print("accuracy",accuracy_score(news_test.target,predicted))
print(metrics.classification_report(news_test.target,predicted,target_names=news_test.target_names))
print("Confusion matrix:\n",metrics.confusion_matrix(news_test.target,predicted))