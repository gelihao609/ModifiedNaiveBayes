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
fitted = TfidfVectorizer(use_idf=False).fit(newsgroups_train.data)
vectors_noIDF_train = fitted.transform(newsgroups_train.data)
vectors_noIDF_test = fitted.transform(newsgroups_test.data)

vectors_train = vectorizer.fit_transform(newsgroups_train.data)
vectors_test = vectorizer.transform(newsgroups_test.data)


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=.01)
clf_noIDF = MultinomialNB(alpha=.01)

clf.fit(vectors_train, newsgroups_train.target)
pred = clf.predict(vectors_test)
print "type of pred: "
print pred.shape
clf_noIDF.fit(vectors_noIDF_train,newsgroups_train.target);
pred_noIDF = clf_noIDF.predict(vectors_noIDF_test)

from sklearn import metrics
# fake = [4 for i in range(1353)]
# pprint(metrics.f1_score(newsgroups_test.target, fake, average='weighted'))

# pprint(metrics.f1_score(newsgroups_test.target, pred, average='weighted'))
# pprint(metrics.f1_score(newsgroups_test.target, pred_noIDF, average='weighted'))

#########Train MNB########
pprint("size of training set: ")
# pprint(type(vectors_train))#scipy.sparse.csr.csr_matrix
pprint(vectors_train.toarray().shape)#
pprint("size of test set: ")
pprint(vectors_test.toarray().shape)

# pprint(type(vectors_train.shape))#tuple
# pprint(type(vectors_train.data.shape))#ndarray

# pprint((vectors_train.shape)[1])#tuple column
# pprint(type(newsgroups_train.target))#numpy.ndarray
# pprint(len(newsgroups_train.target))
# construct TF-IDF Matrix for each class
import numpy as np
# featureNum = (vectors_train.shape)[1]
# sumOfWeight = np.zeros((len(cats),featureNum),dtype=float)
# prior = [0 for i in range(len(cats))]
# vectorArray = vectors_train.toarray()
# print newsgroups_train.target#ndarray with one row
# for row in range(len(newsgroups_train.target)):
# 	# print row
# 	prior[newsgroups_train.target[row]]+=1;
# 	for col in range(featureNum):
# 		sumOfWeight[newsgroups_train.target[row]][col]+= vectorArray[row][col]

# np.save("sumOfWeight", sumOfWeight)
# np.save("prior",prior)
# print prior
# np.save(fname, sumOfWeight)

################conditional probabilities
# sumOfWeight = np.load("sumOfWeight.npy")
# prior = np.load("prior.npy")
# smoothPrameter = 0.01
# sum = [0 for i in range(sumOfWeight.shape[0])]
# for row in range(sumOfWeight.shape[0]):
# 	for col in range(sumOfWeight.shape[1]):
# 		sum[row]+=sumOfWeight[row][col]+smoothPrameter

# conditionalProb = np.zeros((sumOfWeight.shape[0],sumOfWeight.shape[1]),dtype=float)

# for row in range(sumOfWeight.shape[0]):
# 	for col in range(sumOfWeight.shape[1]):
# 		conditionalProb[row][col] = (sumOfWeight[row][col] + smoothPrameter)/sum[row];

# np.save("conditionalProb", conditionalProb)

##################### apply conditional probabilities
### SUM_(TF-IDF_test[i] * log_ConditionalProb[cls][i]) + prior[cls]
prior = np.load("prior.npy")
conditionalProb = np.load("conditionalProb.npy")
testArray = vectors_test.toarray()
score = [float(0) for i in range(conditionalProb.shape[0])]
result = [float(0) for i in range(testArray.shape[0])]
	###add prior

print score

priorSum = 0
for i in range(conditionalProb.shape[0]):
	priorSum+=prior[i]
	
	###add log conditional problibities
for row in range(testArray.shape[0]):
	for i in range(conditionalProb.shape[0]):
		ratio = float(prior[i])/priorSum
		score[i] = np.log(ratio)
	for col in range(testArray.shape[1]):
		for cls in range(conditionalProb.shape[0]):
			score[cls]+= testArray[row][col]*np.log(conditionalProb[cls][col])
	result[row] = score.index(max(score))
	print str(row), str(result[row])
np.save("test_result",result)

###########Metrics############
from sklearn import metrics
import numpy as np
test_result = np.load("test_result.npy")
# pprint(metrics.f1_score(newsgroups_test.target, test_result, average='weighted'))



