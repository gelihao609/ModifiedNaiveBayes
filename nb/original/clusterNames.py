cats = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']
verbose = False;
# cats = ['alt.atheism', 'talk.religion.misc']
import numpy as np
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train', categories=cats,remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', categories=cats,remove=('headers', 'footers', 'quotes'))
from pprint import pprint
pprint("All 4 categories")
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectors_noIDF_train = vectorizer.fit_transform(newsgroups_train.data)
featureNames = vectorizer.get_feature_names()
group9=np.load("/Users/lihao/scikit_learn_data/group9.npy")
# group8=np.load("/Users/lihao/scikit_learn_data/group8.npy")
# group7=np.load("/Users/lihao/scikit_learn_data/group7.npy")
# group6=np.load("/Users/lihao/scikit_learn_data/group6.npy")

group9.resize((len(group9)/2, 2))
# group8.resize((len(group8)/2, 2))
# group7.resize((len(group7)/2, 2))
# group6.resize((len(group6)/2, 2))

import graphDiscover as gd
clusters = gd.getSimilarClusters(group9)
# np.save("/Users/lihao/scikit_learn_data/group8_Clusters_trimmed.npy",clusters) 
flattened_clusters = [y for x in clusters for y in x]
print "num of clusters: " + str(len(clusters))
print "num of features: " + str(len(flattened_clusters))

for index, item in enumerate(clusters):
	print(str(index)+str(item)),
	for i in item:
		print featureNames[int(i)],
	print("")

