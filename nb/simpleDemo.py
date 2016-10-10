# from sklearn.datasets import fetch_20newsgroups
# newsgroups_train = fetch_20newsgroups(subset='train')
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

train_data = ["Chinese Beijing Chinese","Chinese Chinese Shanghai", "Chinese Macao","Tokyo Japan Chinese"]
train_target = [1,1,1,0]
test_data = ["Chinese Chinese Chinese Tokyo Japan"]
vectorizer = TfidfVectorizer()
vectors_train = vectorizer.fit_transform(train_data)
vectors_test = vectorizer.transform(test_data)
vector_train_array = vectors_train.toarray()
featureNum = (vector_train_array.shape)[1]
sumOfWeight = np.zeros((2,featureNum),dtype=float)
prior = [0 for i in range(2)]

for row in range(vector_train_array.shape[0]):
	# print row
	prior[train_target[row]]+=1;
	for col in range(featureNum):
		sumOfWeight[train_target[row]][col]+= vector_train_array[row][col]

smoothPrameter = 0.0001
sum = [0 for i in range(sumOfWeight.shape[0])]
for row in range(sumOfWeight.shape[0]):
	for col in range(sumOfWeight.shape[1]):
		sum[row]+=sumOfWeight[row][col]+smoothPrameter

conditionalProb = np.zeros((sumOfWeight.shape[0],sumOfWeight.shape[1]),dtype=float)

for row in range(sumOfWeight.shape[0]):
	for col in range(sumOfWeight.shape[1]):
		conditionalProb[row][col] = (sumOfWeight[row][col] + smoothPrameter)/sum[row];

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
	score[i] = np.log(ratio)
from pprint import pprint
pprint(sumOfWeight)
pprint(vectorizer.vocabulary_)
pprint(conditionalProb)

print "initial Score:"
print score
	###add log conditional problibities
for row in range(testArray.shape[0]):
	for col in range(testArray.shape[1]):
		for cls in range(conditionalProb.shape[0]):
			score[cls]+= testArray[row][col]*np.log(conditionalProb[cls][col])
			print "current class is " + str(cls) + " current score is: " + str(score[cls])
		print "score for class " + str(cls) +" is: " + str(score[cls])
	print "Get class of max score from: " + str(score)
	result[row] = score.index(max(score))
	print "Max score is in Index " + str(result[row])



