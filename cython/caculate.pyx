import numpy as np
import timeit
def calculate():
	'''
	start = timeit.default_timer()
	similarityMatrix = np.load("/Users/lihao/scikit_learn_data/similarityMatrix.npy")
	vectorArray = np.load("/Users/lihao/scikit_learn_data/vectorArray.npy")
	cols = (vectorArray.shape)[1]
	rows = (vectorArray.shape)[0]
	x = np.matrix(vectorArray)
	y = np.matrix(similarityMatrix)
	vectorArrayWithSimilarity = x*y
	np.save("vectorArrayWithSimilarity",vectorArrayWithSimilarity)
	# for col in range(cols):
	# 	vectorArrayWithSimilarity[0,col] = vectorArray[0,col]
	# 	print "In col: " + str(col) + "/" + str(cols)
	# 	for subCol in range(cols):
	# 		vectorArrayWithSimilarity[0,col] += np.multiply(vectorArray[0,subCol], similarityMatrix[col,subCol])
	end = timeit.default_timer()
	return end - start
	'''
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