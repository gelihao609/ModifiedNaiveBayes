# from sklearn.datasets import fetch_20newsgroups
# newsgroups_train = fetch_20newsgroups(subset='train')
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

train_data = ["Chinese Beijing Chinese","Chinese Chinese Shanghai", "Chinese Macao","Tokyo Japan Chinese"]
train_target = [1,1,1,0]
test_data = ["Chinese Chinese Chinese Tokyo Japan","Chinese Tokyo Japan","Chinese Shanghai Japan Japan","Chinese Macao","Macao Macao Macao Macao"]
vectorizer = CountVectorizer()
vectors_train = vectorizer.fit_transform(train_data)
vectors_test = vectorizer.transform(test_data)
featureNames = vectorizer.get_feature_names()


#######################Add similarity factor ###############
import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = gensim.models.Word2Vec.load_word2vec_format('/Users/lihao/scikit_learn_data/GoogleNews-vectors-negative300.bin', binary=True)  
similarityMatrix = np.zeros((len(featureNames),len(featureNames)),dtype=float)
for i in range(len(featureNames)):
	print "process " + str(i) + "/" + str(len(featureNames))
	if(featureNames[i] in model.vocab):
		for j in range(i, len(featureNames)):
			if(i!=j):
				if(featureNames[j] in model.vocab):
					similarityMatrix[i,j] = model.similarity(featureNames[i],featureNames[j])
					similarityMatrix[j,i] = similarityMatrix[i,j]
#all diagonal are 1
for i in range(similarityMatrix.shape[0]):
	# print "process " + str(i) + "/" + str(similarityMatrix.shape[0]) 
	similarityMatrix[i,i] = 1

vector_train_array = vectors_train.toarray()
vectorArray_matrix = np.matrix(vector_train_array)
similarityMatrix_matrix = np.matrix(similarityMatrix)
vector_train_array = vectorArray_matrix*similarityMatrix_matrix

print "similarityMatrix_matrix"
print similarityMatrix_matrix
print "before"
print vectorArray_matrix
print "after"
print vector_train_array

#######################-----------------------------###############



# vector_train_array = vectors_train.toarray()
featureNum = (vector_train_array.shape)[1]
sumOfWeight = np.zeros((2,featureNum),dtype=float)
prior = [0 for i in range(2)]

# print vectorizer.vocabulary_
# print vector_train_array.shape


for row in range(vector_train_array.shape[0]):
	# print row
	prior[train_target[row]]+=1;
	for col in range(featureNum):
		sumOfWeight[train_target[row],col]+= vector_train_array[row,col]

smoothPrameter = 1
sum = [0 for i in range(sumOfWeight.shape[0])]
for row in range(sumOfWeight.shape[0]):
	for col in range(sumOfWeight.shape[1]):
		sum[row]+=sumOfWeight[row,col]+smoothPrameter

conditionalProb = np.zeros((sumOfWeight.shape[0],sumOfWeight.shape[1]),dtype=float)

for row in range(sumOfWeight.shape[0]):
	for col in range(sumOfWeight.shape[1]):
		conditionalProb[row,col] = (sumOfWeight[row,col] + smoothPrameter)/sum[row];
		print "conditionalProb for " +str(featureNames[col])+ " in class " + str(row) +": " +  str(conditionalProb[row,col])


#########Apply MNB ############
testArray = vectors_test.toarray()
score = [float(0) for i in range(conditionalProb.shape[0])]
result = [float(0) for i in range(testArray.shape[0])]
priorSum = 0
for i in range(conditionalProb.shape[0]):
	priorSum+=prior[i]

	###add prior
for i in range(conditionalProb.shape[0]):
	ratio = float(prior[i])/priorSum
	print "ratio of i: " + str(i) + " is: " + str(ratio)
	score[i] = np.log(ratio)

# from pprint import pprint
# pprint(sumOfWeight)
# pprint(vectorizer.vocabulary_)
# pprint(conditionalProb)

# print "initial Score:"
# print score
# 	###add log conditional problibities
for row in range(testArray.shape[0]):
	for i in range(conditionalProb.shape[0]):
		ratio = float(prior[i])/priorSum
		score[i] = np.log(ratio)
	for col in range(testArray.shape[1]):
		for cls in range(conditionalProb.shape[0]):
			score[cls]+= testArray[row,col]*np.log(conditionalProb[cls,col])
			print "current class is " + str(cls) + " current score is: " + str(score[cls])
	print "Get class of max score from: " + str(score)
	result[row] = score.index(max(score))
	print "Testcase "+ str(row) + " predict " + str(result[row])



