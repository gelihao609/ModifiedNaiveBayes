import numpy as np
group9=np.load("/Users/lihao/scikit_learn_data/group9.npy")
group8=np.load("/Users/lihao/scikit_learn_data/group8.npy")
group9.resize((len(group9)/2, 2))
group8.resize((len(group8)/2, 2))
group89 = np.append(group9,group8,axis=0)
print group89.shape,group8.shape,group9.shape