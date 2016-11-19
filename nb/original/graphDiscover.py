import numpy as np
group9=np.load("/Users/lihao/scikit_learn_data/group9.npy")

group9.resize((len(group9)/2, 2))
# adjList = createAdjList(group9)

def createAdjList(graph):
	adjList = {}
	for i in xrange(graph.shape[0]):
		if str(graph[i,0]) in adjList:
			adjList[str(graph[i,0])].add(graph[i,1])
		else:
			adjList[str(graph[i,0])] = set([graph[i,1]])
		if str(graph[i,1]) in adjList:
			adjList[str(graph[i,1])].add(graph[i,0])
		else:
			adjList[str(graph[i,1])] = set([graph[i,0]])
	return adjList

def bfs(adjList):
	clusters = []
	visited = set()
	for key, value in adjList.iteritems():
		currentCluster = np.array([])
		queue = [key]
		while queue:
			vertex = queue.pop(0)
			if type(vertex) is not str:
			 vertex = str(vertex)
			if vertex not in visited:
				visited.add(vertex)
				currentCluster = np.append(currentCluster,float(vertex))
				queue.extend(list(adjList[vertex]))
		if len(currentCluster.tolist())>0:
			clusters.append(currentCluster.tolist())
	return clusters

def getSimilarClusters(graph):
	return bfs(createAdjList(graph))

def columnMerge(ndarray, clusters, skipIndexs):
	print "skipIndexs: " + str(skipIndexs)
	actualClusters = [i for j, i in enumerate(clusters) if j not in skipIndexs]
	# print actualClusters
	rows = ndarray.shape[0]
	for index, cluster in enumerate(actualClusters):
		# print str(index) + str(cluster)
		newCol = np.zeros((rows))
		merged = sum((ndarray[:,i]) for i in cluster)
		mergedCol = merged.reshape(len(merged),1)
		ndarray = np.column_stack((ndarray,mergedCol))
		# print ndarray 
	flattened_clusters = [y for x in actualClusters for y in x]
	flattened_clusters.sort()
	ndarray = np.delete(ndarray, flattened_clusters, 1)
	return ndarray

