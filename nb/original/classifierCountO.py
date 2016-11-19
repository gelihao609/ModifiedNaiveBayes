cats = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']
verbose = False;
# cats = ['alt.atheism', 'talk.religion.misc']
import numpy as np
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train', categories=cats,remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', categories=cats,remove=('headers', 'footers', 'quotes'))
from pprint import pprint
pprint("All 4 categories")
#######################  Stop words ##################
print("stop word")
import stopword as sw
nostop_newsgroups_train = sw.removeSW(newsgroups_train.data)
nostop_newsgroups_test = sw.removeSW(newsgroups_test.data)
# import similarityMaker as sm
# sm.makeSimilarityMatrixToFile(stemmed_newsgroups_train)
print nostop_newsgroups_train
print("finish stop word")
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectors_noIDF_train = vectorizer.fit_transform(nostop_newsgroups_train)
vectors_noIDF_test = vectorizer.transform(nostop_newsgroups_test)
pprint(vectors_noIDF_train.shape) #2034 
pprint(vectors_noIDF_test.shape) #

########################  Stemming ###################


pprint("stemming")
import stemming
stemmed_newsgroups_train = stemming.stem(newsgroups_train.data)
stemmed_newsgroups_test = stemming.stem(newsgroups_test.data)
pprint("finish stemming")


########################  ###################
pprint("vectorize")
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
# vectors_noIDF_train = vectorizer.fit_transform(newsgroups_train.data)
# vectors_noIDF_test = vectorizer.transform(newsgroups_test.data)
# pprint(vectors_noIDF_train.shape) #2034 
# pprint(vectors_noIDF_test.shape) #1353 
vectors_noIDF_train = vectorizer.fit_transform(stemmed_newsgroups_train)
vectors_noIDF_test = vectorizer.transform(stemmed_newsgroups_test)


########################  calculate vector weight with similarity ###################
# print "load similarity"
# vectorMatrixWithSimilarity = np.load("/Users/lihao/scikit_learn_data/vectorMatrixNoNegWithSimilarity_diag0.npy")
# # print "Max: " + str(vectorMatrixWithSimilarity.max())
# # print "Min: " + str(vectorMatrixWithSimilarity.min())
# print vectorMatrixWithSimilarity.shape
# print vectors_noIDF_train.toarray().shape

# scalable_factor = 0.1
# vectorArray = np.matrix(vectors_noIDF_train.toarray()) + scalable_factor * vectorMatrixWithSimilarity
# vectors_noIDF_train = vectorArray

# ######################## present features that have high similarities ###################
# group9=np.load("/Users/lihao/scikit_learn_data/group9.npy")
# group8=np.load("/Users/lihao/scikit_learn_data/group8.npy")
# group7=np.load("/Users/lihao/scikit_learn_data/group7.npy")
# group6=np.load("/Users/lihao/scikit_learn_data/group6.npy")

group9=np.load("/Users/lihao/scikit_learn_data/group9_stem.npy")
group8=np.load("/Users/lihao/scikit_learn_data/group8_stem.npy")
group7=np.load("/Users/lihao/scikit_learn_data/group7_stem.npy")
group6=np.load("/Users/lihao/scikit_learn_data/group6_stem.npy")


group9.resize((len(group9)/2, 2))
group8.resize((len(group8)/2, 2))
group7.resize((len(group7)/2, 2))
group6.resize((len(group6)/2, 2))

# group89 = np.append(group9,group8, axis=0)
# group789 = np.append(group89,group7, axis=0)
# group6789 = np.append(group789,group6, axis=0)

import graphDiscover as gd
clusters = gd.getSimilarClusters(group6789)
# skip = [0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,18,19,21,22,23,24,25,26]
# skip = range(0,100)
# skip1 = range(100,200)
# skip2 = range(200,300)
# skip3 = range(300,400)
# skip4 = range(400,500)
skip = []
# np.save("/Users/lihao/scikit_learn_data/group6_Clusters.npy",clusters)
# clusters=np.load("/Users/lihao/scikit_learn_data/group8_Clusters_trimmed.npy")
# flattened_clusters = [y for x in clusters for y in x]
# print clusters.shape
# print len(flattened_clusters)
from sklearn.naive_bayes import MultinomialNB
merged_train = gd.columnMerge(vectors_noIDF_train.toarray(),clusters,skip)
merged_test = gd.columnMerge(vectors_noIDF_test.toarray(),clusters,skip)

print merged_train

clf_noIDF = MultinomialNB(alpha=.01)
# clf_noIDF.fit(vectors_noIDF_train, newsgroups_train.target)
# pred = clf_noIDF.predict(vectors_noIDF_test)
clf_noIDF.fit(merged_train, newsgroups_train.target)
pred = clf_noIDF.predict(merged_test)

from sklearn import metrics
print "Original count, " + "categories number is: " + str(len(cats))
pprint(metrics.f1_score(newsgroups_test.target, pred, average='weighted'))


# np.save("/Users/lihao/scikit_learn_data/group8_merged_train_trimmed.npy",merged_train)
# np.save("/Users/lihao/scikit_learn_data/group8_merged_test_trimmed.npy",merged_test)
# group8_merged_train=np.load("/Users/lihao/scikit_learn_data/group8_merged_train_trimmed.npy")
# group8_merged_test=np.load("/Users/lihao/scikit_learn_data/group8_merged_test_trimmed.npy")
# group7_merged_train=np.load("/Users/lihao/scikit_learn_data/group7_merged_train.npy")
# group7_merged_test=np.load("/Users/lihao/scikit_learn_data/group7_merged_test.npy")
# group6_merged_train=np.load("/Users/lihao/scikit_learn_data/group6_merged_train.npy")
# group6_merged_test=np.load("/Users/lihao/scikit_learn_data/group6_merged_test.npy")

# group789_merged_train=np.load("/Users/lihao/scikit_learn_data/group789_merged_train.npy")
# group789_merged_test=np.load("/Users/lihao/scikit_learn_data/group789_merged_test.npy")

# assert (group8_merged_train.shape[1] == group8_merged_test.shape[1]),"should be same column!"
# assert (group8_merged_train.shape[1] == vectors_noIDF_train.shape[1]+len(clusters)-len(flattened_clusters)),"should be equal!"

# print group6_merged_train.shape,group6_merged_test.shape
# ######################## get new vector from mutiply similarities ###################
# vectorMatrixTestWithSimilarity = np.load("/Users/lihao/scikit_learn_data/vectorMatrixTestNoNegWithSimilarity_diag0.npy")
# print vectorMatrixTestWithSimilarity.shape
# print vectors_noIDF_test.toarray().shape
# vectorArray_test = np.matrix(vectors_noIDF_test.toarray()) + scalable_factor * vectorMatrixTestWithSimilarity
# vectors_noIDF_test = vectorArray_test
# ########################  ###################


# clf_noIDF.fit(vectors_noIDF_train, newsgroups_train.target)
# pred = clf_noIDF.predict(vectors_noIDF_test)
'''


