import numpy as np
similarity = np.load("/Users/lihao/scikit_learn_data/similarityMatrix_stem.npy")
positiveCount = 0
negativeCount = 0
zeroCount = 0
oneCount = 0
largerThanOne = 0
eightCount = 0
sixCount = 0
zeroTosixCount = 0

# for i in xrange(similarity.shape[1]):
# 	for j in xrange(i, similarity.shape[1]):
# 		if(similarity[i,j]==0):
# 			continue
# 		elif(similarity[i,j]>0.8 and similarity[i,j]<1):
# 			# print str(i) + " " +str(j) + ": " + str(similarity[i,j])
# 			positiveCount+=1
# 			print ">0.8: " + str(i)+" "+str(j)+" "str(positiveCount)
# 		elif(similarity[i,j]>=0.6 and similarity[i,j]<0.8):
# 			zeroCount+=1
# 			print "zeroCount: " + str(zeroCount)
# 		elif(similarity[i,j]==1):
# 			oneCount+=1
# 			print "oneCount: " + str(oneCount)
# 		elif(similarity[i,j]<0):
# 			similarity[i,j] = 0
# 			negativeCount+=1
# 			print "negativeCount: " + str(i)+" "+str(j)+" "+str(negativeCount)
# 		elif(similarity[i,j]>1):
# 			largerThanOne+=1
# 			print "largerThanOne: " + str(largerThanOne)
# 		else:
# 			print "(0.0.6): " + str(zeroTosixCount)
# 			zeroTosixCount+=1

# print "Final: " + " positiveCount: " + str(positiveCount) + " zeroCount: " + str(zeroCount) +" oneCount: " + str(oneCount) +" negativeCount: " + str(negativeCount) + " largerThanOne: " + str(largerThanOne)
# print "max: " + str(similarity.max())
# print "min: " + str(similarity.min())

###############find similarities in matrix ###############
# group9 = np.array([]) 
# group8 = np.array([]) 
# group7 = np.array([]) 
# group6 = np.array([]) 

# for i in xrange(similarity.shape[1]):
# 	for j in xrange(i, similarity.shape[1]):
# 		if(similarity[i,j]>=0.9 and similarity[i,j]<1):
# 			print ">=0.9: i="+str(i)+" j="+str(j)
# 			group9 = np.append(group9,[i,j])
# 		elif(similarity[i,j]>=0.8 and similarity[i,j]<0.9):
# 			print ">=0.8: i="+str(i)+" j="+str(j)
# 			group8 = np.append(group8,[i,j])
# 		elif(similarity[i,j]>=0.7 and similarity[i,j]<0.8):
# 			print ">=0.7: i="+str(i)+" j="+str(j)
# 			group7 = np.append(group7,[i,j])
# 		elif(similarity[i,j]>=0.6 and similarity[i,j]<0.7):
# 			print ">=0.6: i="+str(i)+" j="+str(j)
# 			group6 = np.append(group6,[i,j])

# np.save("/Users/lihao/scikit_learn_data/group9_stem",group9)
# np.save("/Users/lihao/scikit_learn_data/group8_stem",group8)
# np.save("/Users/lihao/scikit_learn_data/group7_stem",group7)
# np.save("/Users/lihao/scikit_learn_data/group6_stem",group6)

###############find to merge features ###############
# group9=np.load("/Users/lihao/scikit_learn_data/group9_stem.npy")
# group8=np.load("/Users/lihao/scikit_learn_data/group8_stem.npy")
# group7=np.load("/Users/lihao/scikit_learn_data/group7_stem.npy")
# group6=np.load("/Users/lihao/scikit_learn_data/group6_stem.npy")
# group9=np.load("/Users/lihao/scikit_learn_data/group9.npy")
# group8=np.load("/Users/lihao/scikit_learn_data/group8.npy")
# group7=np.load("/Users/lihao/scikit_learn_data/group7.npy")
# group6=np.load("/Users/lihao/scikit_learn_data/group6.npy")

# group9.resize(len(group9)/2,2)
# print len(group9), len(group8),len(group7),len(group6)


