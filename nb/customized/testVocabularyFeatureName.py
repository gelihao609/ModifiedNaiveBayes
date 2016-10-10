from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = gensim.models.Word2Vec.load_word2vec_format('./dataset/GoogleNews-vectors-negative300.bin', binary=True)  
vocab_word2vec = model.vocab.keys()
train_data = ["Chinese Beijing Chinese","Chinese Chinese Shanghai", "Chinese Macao","Tokyo Japan Chinese"]
vectorizer = CountVectorizer()
vectors_train = vectorizer.fit_transform(train_data)
featureNames = vectorizer.get_feature_names()
similarityMatrix = np.zeros((len(featureNames),len(featureNames)),dtype=float)
for i in range(len(featureNames)):
	print "process " + str(i) + "/" + str(len(featureNames))
	if(featureNames[i] in model.vocab):
		for j in range(len(featureNames)):
			if(i!=j):
				if(featureNames[j] in model.vocab):
					similarityMatrix[i][j] = model.similarity(featureNames[i],featureNames[j])
					# print "similarity for "+str(featureNames[i]) +" " + str(featureNames[j]) + " is: " + str(similarityMatrix[i][j])
	# else:
	# 	print str(featureNames[i]) + "is not in vocab"		