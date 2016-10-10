cats = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']
from sklearn.datasets import fetch_20newsgroups
# alt_atheism_train = fetch_20newsgroups(subset='train', categories=['alt.atheism'],remove=('headers', 'footers', 'quotes'))
# alt_atheism_test = fetch_20newsgroups(subset='test', categories=['alt.atheism'],remove=('headers', 'footers', 'quotes'))
# talk_religion_misc_train = fetch_20newsgroups(subset='train', categories=['talk.religion.misc'],remove=('headers', 'footers', 'quotes'))
# talk_religion_misc_test = fetch_20newsgroups(subset='test', categories=['talk.religion.misc'],remove=('headers', 'footers', 'quotes'))
# comp_graphics_train = fetch_20newsgroups(subset='train', categories=['comp.graphics'],remove=('headers', 'footers', 'quotes'))
# comp_graphics_test = fetch_20newsgroups(subset='test', categories=['comp.graphics'],remove=('headers', 'footers', 'quotes'))
# sci_space_train = fetch_20newsgroups(subset='train', categories=['sci.space'],remove=('headers', 'footers', 'quotes'))
# sci_space_test = fetch_20newsgroups(subset='test', categories=['sci.space'],remove=('headers', 'footers', 'quotes'))

newsgroups_train = fetch_20newsgroups(subset='train', categories=cats,remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', categories=cats,remove=('headers', 'footers', 'quotes'))
from pprint import pprint
# pprint("alt_atheism")
# pprint(alt_atheism_train.target.shape) #480 
# pprint(alt_atheism_test.target.shape) #319
# pprint("talk_religion_misc")
# pprint(talk_religion_misc_train.target.shape) #377
# pprint(talk_religion_misc_test.target.shape) #251
# pprint("comp_graphics")
# pprint(comp_graphics_train.target.shape) #584 
# pprint(comp_graphics_test.target.shape) #389
# pprint("sci_space")
# pprint(sci_space_train.target.shape) #593 
# pprint(sci_space_test.target.shape) #394
pprint("All 4 categories")
pprint(newsgroups_train.target.shape) #2034 
pprint(newsgroups_test.target.shape) #1353
from sklearn.feature_extraction.text import TfidfVectorizer
# pprint(type(newsgroups_train.data[0]))#list --> 'unicode'
# pprint((newsgroups_train.data[0]))#list --> 'unicode'

vectorizer = TfidfVectorizer()

# fitted = TfidfVectorizer(use_idf=False).fit(newsgroups_train.data)
# vectors_noIDF_train = fitted.transform(newsgroups_train.data)
# vectors_noIDF_test = fitted.transform(newsgroups_test.data)

vectors_train = vectorizer.fit_transform(newsgroups_train.data)
vectors_test = vectorizer.transform(newsgroups_test.data)


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=.01)
# clf_noIDF = MultinomialNB(alpha=.01)

clf.fit(vectors_train, newsgroups_train.target)
pred = clf.predict(vectors_test)

# clf_noIDF.fit(vectors_noIDF_train,newsgroups_train.target);
# pred_noIDF = clf_noIDF.predict(vectors_noIDF_test)

from sklearn import metrics
print "With IDF, " + "categories number is: " + str(len(cats))
pprint(metrics.f1_score(newsgroups_test.target, pred, average='weighted'))
# print "No IDF, " + "categories number is: " + len(cats)
# pprint(metrics.f1_score(newsgroups_test.target, pred_noIDF, average='weighted'))




