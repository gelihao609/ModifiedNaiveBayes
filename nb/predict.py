from sklearn import metrics
pprint(metrics.f1_score(newsgroups_test.target, pred, average='weighted'))
import numpy as np
test_result = np.load("test_result.npy")
