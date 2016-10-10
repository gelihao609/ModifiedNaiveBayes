print "customized NB..."
#########################  dataset ###################################
cats = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']
# cats = ['alt.atheism', 'talk.religion.misc']
print "categories: " + str(cats)
verbose = False;
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train', categories=cats,remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', categories=cats,remove=('headers', 'footers', 'quotes'))
from pprint import pprint
print "training set: " + str(newsgroups_train.target.shape)
print "testing set: " + str(newsgroups_test.target.shape) #1353


########################  tokenization ####################################
print "tokenization with TfidfVectorizer..."
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectors_noIDF_train = vectorizer.fit_transform(newsgroups_train.data)
vectors_noIDF_test = vectorizer.transform(newsgroups_test.data)

########################  calculate total token weight of each class ####################################
print "calculate total token weight of each class..."
import numpy as np
featureNum = (vectors_noIDF_train.shape)[1]
sumOfWeight = np.zeros((len(cats),featureNum),dtype=float)
prior = [0 for i in range(len(cats))]
vectorArray = vectors_noIDF_train.toarray()

for row in range(len(newsgroups_train.target)):
	# print row
	prior[newsgroups_train.target[row]]+=1;
	if(verbose):
		print "In row: "+str(row)+ ", Sum up word count for class: " + str(newsgroups_train.target[row]) 
	for col in range(featureNum):
		sumOfWeight[newsgroups_train.target[row]][col]+= vectorArray[row][col]

# np.save("sumOfWeightCount", sumOfWeight)
# np.save("priorCount",prior)
# print prior
# np.save(fname, sumOfWeight)

########################  calculate conditional probability ####################################
print "calculate conditional probability..."
smoothPrameter = 0.01
print "smoothPrameter is: " + str(smoothPrameter)
summ = [0 for i in range(sumOfWeight.shape[0])]
for row in range(sumOfWeight.shape[0]):
	for col in range(sumOfWeight.shape[1]):
		summ[row]+=sumOfWeight[row][col]+smoothPrameter

conditionalProb = np.zeros((sumOfWeight.shape[0],sumOfWeight.shape[1]),dtype=float)

for row in range(sumOfWeight.shape[0]):
	for col in range(sumOfWeight.shape[1]):
		conditionalProb[row][col] = (sumOfWeight[row][col] + smoothPrameter)/summ[row];


########################  Apply NB on test texts ####################################
print "Apply NB on test texts..."
testArray = vectors_noIDF_test.toarray()

score = [float(0) for i in range(conditionalProb.shape[0])]
result = [float(0) for i in range(testArray.shape[0])]

priorSum = 0
for i in range(conditionalProb.shape[0]):
	priorSum+=prior[i]

	###add log conditional problibities
for row in range(testArray.shape[0]):
	for i in range(conditionalProb.shape[0]):
		ratio = float(prior[i])/priorSum
		score[i] = np.log(ratio)
	for cls in range(conditionalProb.shape[0]):
		for col in range(testArray.shape[1]):
			score[cls]+= testArray[row][col]*np.log(conditionalProb[cls][col])
		if(verbose):
			print "Number of test: " + str(row) + ", Calculate class " + str(cls) + ", score is: " + str(score[cls])
	result[row] = score.index(max(score))
	if(verbose):
		print "Num " + str(row) +"/"+ str(testArray.shape[0]) +" Predict class index: " + str(result[row]) + "; Target is: " + str(newsgroups_test.target[row]);

########################  Measurement  ####################################
from sklearn import metrics
print "Measurement..."
pprint(metrics.f1_score(newsgroups_test.target, result, average='weighted'))




