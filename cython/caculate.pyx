import numpy as np
import timeit
def calculate():
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