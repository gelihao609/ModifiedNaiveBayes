import numpy as np
array = np.zeros((4,10),dtype=float)
# print type(array[0][0])
# for(int j=0;j<row.length;j++)
# {
# 	for(int i=0;i<column.length;i++)
# 	{
# 		array[i][dataset_train.data.target(j)]+=dataset_train.data.getRow(j).getCol(i);
# 	}
# }
for x in range(array.shape[0]):	
	for y in range(array.shape[1]):
		print x,y