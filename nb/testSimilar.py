from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
# train_data = ["Chinese Beijing Chinese","Chinese Chinese Shanghai", "Chinese Macao","Tokyo Japan Chinese"]
# vectors_train = vectorizer.fit_transform(train_data)

cats = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']
verbose = False;
# cats = ['alt.atheism', 'talk.religion.misc']

# from sklearn.datasets import fetch_20newsgroups
# newsgroups_train = fetch_20newsgroups(subset='train', categories=cats,remove=('headers', 'footers', 'quotes'))

# vectorizer = CountVectorizer()
# vectors_train = vectorizer.fit_transform(newsgroups_train.data)

# import gensim
# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# model = gensim.models.Word2Vec.load_word2vec_format('/Users/lihao/scikit_learn_data/GoogleNews-vectors-negative300.bin', binary=True)  
# featureNames = vectorizer.get_feature_names()
# i = 5631
# j = 20697
# print featureNames[i], featureNames[j]
# print model.similarity(featureNames[i],featureNames[j])

# i = 5631
# j = 20596
# print featureNames[i], featureNames[j]
# print model.similarity(featureNames[i],featureNames[j])

# i = 5773
# j = 11763
# print featureNames[i], featureNames[j]
# print model.similarity(featureNames[i],featureNames[j])

# i = 7516
# j = 19161
# print featureNames[i], featureNames[j]
# print model.similarity(featureNames[i],featureNames[j])


# similarityMatrix = np.zeros((len(featureNames),len(featureNames)),dtype=float)
# for i in range(len(featureNames)):
# 	print "process " + str(i) + "/" + str(len(featureNames))
# 	if(featureNames[i] in model.vocab):
# 		for j in range(i, len(featureNames)):
# 			if(i!=j):
# 				if(featureNames[j] in model.vocab):
# 					similarityMatrix[i][j] = model.similarity(featureNames[i],featureNames[j])
# 					similarityMatrix[j][i] = similarityMatrix[i][j]
# 			else:
# 				similarityMatrix[i][j] = 1;

# print similarityMatrix
# np.save("similarityMatrix_demo", similarityMatrix)
# similarityMatrix = np.load("/Users/lihao/scikit_learn_data/similarityMatrix.npy")
# vectorMatrixWithSimilarity = np.load("/Users/lihao/scikit_learn_data/vectorMatrixWithSimilarity.npy")


# 	similarityMatrix[i][i] = 1
# np.save("/Users/lihao/scikit_learn_data/similarityMatrix_diagnoal1",similarityMatrix)


