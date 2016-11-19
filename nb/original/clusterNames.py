cats = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']
verbose = False;
# cats = ['alt.atheism', 'talk.religion.misc']
import numpy as np
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train', categories=cats,remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', categories=cats,remove=('headers', 'footers', 'quotes'))
from pprint import pprint
pprint("All 4 categories")
import stemming
stemmed_newsgroups_train = stemming.stem(newsgroups_train.data)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectors_noIDF_train = vectorizer.fit_transform(stemmed_newsgroups_train)
featureNames = vectorizer.get_feature_names()
group9=np.load("/Users/lihao/scikit_learn_data/group9_stem.npy")
group8=np.load("/Users/lihao/scikit_learn_data/group8_stem.npy")
group7=np.load("/Users/lihao/scikit_learn_data/group7_stem.npy")
group6=np.load("/Users/lihao/scikit_learn_data/group6_stem.npy")

group9.resize((len(group9)/2, 2))
group8.resize((len(group8)/2, 2))
group7.resize((len(group7)/2, 2))
group6.resize((len(group6)/2, 2))

group89 = np.append(group9,group8, axis=0)
group789 = np.append(group89,group7, axis=0)
group6789 = np.append(group789,group6, axis=0)

import graphDiscover as gd
clusters = gd.getSimilarClusters(group6789)
# np.save("/Users/lihao/scikit_learn_data/group8_Clusters_trimmed.npy",clusters) 
flattened_clusters = [y for x in clusters for y in x]
print "num of clusters: " + str(len(clusters))
print "num of features: " + str(len(flattened_clusters))

for index, item in enumerate(clusters):
	print(str(index)+str(item)),
	for i in item:
		print featureNames[int(i)],
	print("")

