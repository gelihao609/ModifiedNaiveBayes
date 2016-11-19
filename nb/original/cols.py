import numpy as np
a = np.array([[1, 2,0,3,1,5,4], [3, 4,0,5,1,1,4]])
b = np.zeros((a.shape[0],1))
c = a[:,1]
d = c + a[:,1]
e = np.column_stack((a,d))


# import graphDiscover as gd

# clusters = [[1,4,2],[5,6]]
# cl = np.array(clusters)
# # print cl.flatten()
# f = gd.columnMerge(a,clusters) 

# print cl
# list_of_lists = [[1,3,2], [0,4]]
# flattened_list = [y for x in list_of_lists for y in x]
# a = np.delete(a,[0,1,2,3],1);
# print a


# fil = [0,1,3]
# list_of_lists = [[1,3,2], [0,4],[1,3],[1,1,1],[2,2,2]]
# # flattened_list = [y for x in list_of_lists for y in x]
# # a = np.delete(a,[0,1,2,3],1);
# for index, item in enumerate(list_of_lists):
# 	if(index not in fil):
# 		print item

# a = np.arange(16).reshape(4,4)
# b = [1,2]
# print a 
# c = sum((a[:,i]) for i in b)
# d = c.reshape(len(c),1)
# e = np.column_stack((a,d))

b = [range(6)]
print b