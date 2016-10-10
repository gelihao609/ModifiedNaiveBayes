cats = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']
verbose = False;
# cats = ['alt.atheism', 'talk.religion.misc']

from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train', categories=cats,remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', categories=cats,remove=('headers', 'footers', 'quotes'))
from pprint import pprint
pprint("All 4 categories")
pprint(newsgroups_train.target.shape) #2034 
pprint(newsgroups_test.target.shape) #1353
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectors_noIDF_train = vectorizer.fit_transform(newsgroups_train.data)
vectors_noIDF_test = vectorizer.transform(newsgroups_test.data)

from sklearn.naive_bayes import MultinomialNB
clf_noIDF = MultinomialNB(alpha=.01)

clf_noIDF.fit(vectors_noIDF_train, newsgroups_train.target)
pred = clf_noIDF.predict(vectors_noIDF_test)

from sklearn import metrics
print "Original count, " + "categories number is: " + str(len(cats))
pprint(metrics.f1_score(newsgroups_test.target, pred, average='weighted'))


