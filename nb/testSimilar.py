from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
# train_data = ["Chinese Beijing Chinese","Chinese Chinese Shanghai", "Chinese Macao","Tokyo Japan Chinese"]
# vectorizer = CountVectorizer()
# vectors_train = vectorizer.fit_transform(train_data)

# import gensim
# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# model = gensim.models.Word2Vec.load_word2vec_format('/Users/lihao/scikit_learn_data/GoogleNews-vectors-negative300.bin', binary=True)  
# featureNames = vectorizer.get_feature_names()
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
similarityMatrix = np.load("/Users/lihao/scikit_learn_data/similarityMatrix.npy")

for i in range(similarityMatrix.shape[0]):
	print "process " + str(i) + "/" + str(similarityMatrix.shape[0]) 
	similarityMatrix[i][i] = 1
np.save("/Users/lihao/scikit_learn_data/similarityMatrix_diagnoal1",similarityMatrix)