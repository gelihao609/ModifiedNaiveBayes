import numpy as np
from scipy import sparse
# x = np.matrix( ((2,3), (0, 5), (0, 5),(0, 5), (0, 5),(0, 5), (0, 5),(0,0),(0,0),(0,0),(0,0),(0,0)
# 	, (0, 5), (0, 5),(0, 5), (0, 5),(0, 5), (0, 5),(0,0),(0,0),(0,0),(0,0),(0,0)) )
# y = np.matrix( ((1,1,0,0,0,0,0,2,0,2,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,2,0,2,1), 
# 	(0,0,0,0,0,5,0, -1,0,2,1,0,0,0,1,0,0,0,1,1,0,0,0,0,0,2,0,2,1)) )
x = np.matrix( ((2,3), (0, 5)))
y = np.matrix(((1,3,1),(5,1,1)))
# sX = sparse.csr_matrix(x)
# sY = sparse.csr_matrix(y)
# a = sX * sY
# a = x * y
# # np.save("testMatrix",a);
# a = np.load("testMatrix.npy");
# print a

