print "customized NB..."

#########################  dataset ###################################
import numpy as np
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
print "tokenization with CountVectorizer..."
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectors_noIDF_train = vectorizer.fit_transform(newsgroups_train.data)
vectors_noIDF_test = vectorizer.transform(newsgroups_test.data)
# np.save("vectorArray", vectors_noIDF_train.toarray())
# np.save("vectors_noIDF_test", vectors_noIDF_test)
###################### calculate similarity of each word-pair #####################
'''
print "calculate similarity from word2vec..."
import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = gensim.models.Word2Vec.load_word2vec_format('/Users/lihao/scikit_learn_data/GoogleNews-vectors-negative300.bin', binary=True)  
featureNames = vectorizer.get_feature_names()
similarityMatrix = np.zeros((len(featureNames),len(featureNames)),dtype=float)
for i in range(len(featureNames)):
	print "process " + str(i) + "/" + str(len(featureNames))
	if(featureNames[i] in model.vocab):
		for j in range(i, len(featureNames)):
			if(i!=j):
				if(featureNames[j] in model.vocab):
					s = model.similarity(featureNames[i],featureNames[j])
					if (s > 0):
						similarityMatrix[i,j] = s
						similarityMatrix[j,i] = similarityMatrix[i,j]
# all diagonal are 1, avoid cancel weight when do matrix multiply
for i in range(similarityMatrix.shape[0]):
	print "process " + str(i) + "/" + str(similarityMatrix.shape[0]) 
	similarityMatrix[i][i] = 1

np.save("/Users/lihao/scikit_learn_data/similarityMatrixNoNeg", similarityMatrix)

 			#######exclude eslf #####
similarityMatrix = np.load("/Users/lihao/scikit_learn_data/similarityMatrixNoNeg.npy")
for i in range(similarityMatrix.shape[0]):
	print "process " + str(i) + "/" + str(similarityMatrix.shape[0]) 
	similarityMatrix[i][i] = 0
np.save("/Users/lihao/scikit_learn_data/similarityMatrixNoNeg_diag0", similarityMatrix)
'''

################### calculate weight with similarity ###################

print "calculate weight with similarity..."
similarityMatrix = np.load("/Users/lihao/scikit_learn_data/similarityMatrix_diagnoal1.npy")
# vectorArray_matrix = np.matrix(vectors_noIDF_train.toarray())
vectorArray_matrix = np.matrix(vectors_noIDF_test.toarray())

similarityMatrix_matrix = np.matrix(similarityMatrix)
result = vectorArray_matrix * similarityMatrix_matrix
np.save("/Users/lihao/scikit_learn_data/vectorMatrixTestNoNegWithSimilarity_diag1", result)

########################  calculate total token weight of each class ###################
'''
print "calculate total token weight of each class..."
featureNum = (vectors_noIDF_train.shape)[1]
sumOfWeight = np.zeros((len(cats),featureNum),dtype=float)
prior = [0 for i in range(len(cats))]
vectorMatrixWithSimilarity = np.load("/Users/lihao/scikit_learn_data/similarityMatrixNoNeg.npy")
scalable_factor = 0
vectorArray = np.matrix(vectors_noIDF_train.toarray()) + scalable_factor * vectorMatrixWithSimilarity
########################  decreate sample size for DEBUG ###################
testArray = vectors_noIDF_test.toarray()
# DEBUG = False
DEBUG = True
if(DEBUG):
	row_Debug = 10
	col_Debug = 26879
	vectorArray = vectorArray[:row_Debug,:col_Debug]
	newsgroups_train.target = newsgroups_train.target[:row_Debug]
	newsgroups_test.target = newsgroups_test.target[:row_Debug]
	testArray = testArray[:row_Debug,:col_Debug]
	featureNum = col_Debug

print "trained vector shape:" + str(vectorArray.shape)
print "trained target shape:" + str(newsgroups_train.target.shape)
print "test vector shape:" + str(testArray.shape)

########################  DEBUG DEBUG DEBUG DEBUG DEBUG ###################

for row in range(len(newsgroups_train.target)):
	# print row
	prior[newsgroups_train.target[row]]+=1
	if(verbose):
		print "In row: "+str(row)+ ", Sum up word count for class: " + str(newsgroups_train.target[row]) 
	for col in range(featureNum):
		sumOfWeight[newsgroups_train.target[row],col]+= vectorArray[row,col]

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
		summ[row]+=sumOfWeight[row, col]+smoothPrameter

conditionalProb = np.zeros((sumOfWeight.shape[0],sumOfWeight.shape[1]),dtype=float)

for row in range(sumOfWeight.shape[0]):
	for col in range(sumOfWeight.shape[1]):
		conditionalProb[row,col] = (sumOfWeight[row,col] + smoothPrameter)/summ[row];


########################  Apply NB on test texts ####################################
print "Apply NB on test texts..."

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
			score[cls]+= testArray[row,col]*np.log(conditionalProb[cls,col])
		if(verbose):
			print "Number of test: " + str(row) + ", Calculate class " + str(cls) + ", score is: " + str(score[cls])
	result[row] = score.index(max(score))
	print "Num " + str(row) +"/"+ str(testArray.shape[0]) +" Predict class index: " + str(result[row]) + "; Target is: " + str(newsgroups_test.target[row]);

########################  Measurement  ####################################
from sklearn import metrics
print "Measurement..."
pprint(metrics.f1_score(newsgroups_test.target, result, average='weighted'))
'''