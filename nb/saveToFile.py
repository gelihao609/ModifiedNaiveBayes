import numpy as np
sumOfWeight = np.zeros((4,10),dtype=float)
# np.save("sumOfWeight", sumOfWeight)
imported = np.load("prior" + '.npy')
print imported