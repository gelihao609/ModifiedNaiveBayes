import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
train_data = ["He likes to eat apple","we are fond of having noodle", "He likes to play football","He plays tennis"]
train_target = [1,1,0,0]

test_data = ["She want to get pizza", "we would like to dance","She likes to have apple"]
test_target = [1,0,1]
	
vectorizer = CountVectorizer()
vectors_train = vectorizer.fit_transform(train_data)
vectors_test = vectorizer.transform(test_data)

from sklearn.naive_bayes import MultinomialNB
clf_noIDF = MultinomialNB(alpha=.01)
clf_noIDF.fit(vectors_train, train_target)
pred = clf_noIDF.predict(vectors_test)

print vectorizer.get_feature_names()
print "vectors_train: " + str(vectors_train.toarray())
print "vectors_test: " + str(vectors_test.toarray())

print "original pred: " + str(pred)

############## get similarity matrix ###########

import similarityMaker as sm
similarityMatrix = np.load("/Users/lihao/scikit_learn_data/easyCheck.npy");
# similarityMatrix = sm.makeSimilarityMatrixToFile(train_data, "easyCheck")

print str(similarityMatrix)

group = sm.getSimilarityPairsLargerThan(similarityMatrix,0.3)
group.resize((len(group)/2, 2))

import graphDiscover as gd
clusters = gd.getSimilarClusters(group)

print  "Clusters: "
featureNames = vectorizer.get_feature_names()
for index, item in enumerate(clusters):
	print(str(index)+str(item)),
	for i in item:
		print featureNames[int(i)],
	print("")
skip = []
merged_train = gd.columnMerge(vectors_train.toarray(),clusters,skip)
merged_test = gd.columnMerge(vectors_test.toarray(),clusters,skip)

print "merged_train: " + str(merged_train)
print "merged_test: " + str(merged_test)

clf_merged = MultinomialNB(alpha=.01)

clf_merged.fit(merged_train, train_target)
pred_merged = clf_merged.predict(merged_test)

print "Merged pred:" + str(pred_merged)


