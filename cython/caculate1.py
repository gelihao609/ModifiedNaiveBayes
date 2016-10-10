import numpy as np
import timeit
from scipy import sparse
def calculate():
	start = timeit.default_timer()
	similarityMatrix = np.load("/Users/lihao/scikit_learn_data/similarityMatrix_diagnoal1.npy")
	vectorArray = np.load("/Users/lihao/scikit_learn_data/vectorArray.npy")
	# similarityMatrix = similarityMatrix[:10000,:10000]
	# vectorArray = vectorArray[:,:10000]

	vectorArray_matrix = np.matrix(vectorArray)
	similarityMatrix_matrix = np.matrix(similarityMatrix)
	result = vectorArray_matrix * similarityMatrix_matrix

	# ssX = sparse.csr_matrix(vectorArray)
	# sparseSM = sparse.csr_matrix(similarityMatrix)
	# result = ssX * sparseSM

	# print sparseSM.shape, sparseSM[0,0], sparseSM[26876, 26878]
	# for col in range(cols):
	# 	vectorArrayWithSimilarity[0,col] = vectorArray[0,col]
	# 	print "In col: " + str(col) + "/" + str(cols)
	# 	for subCol in range(cols):
	# 		vectorArrayWithSimilarity[0,col] += np.multiply(vectorArray[0,col], similarityMatrix[col,subCol])
	print result.shape
	np.save("vectorMatrixWithSimilarity",result)
	end = timeit.default_timer()
	return end - start