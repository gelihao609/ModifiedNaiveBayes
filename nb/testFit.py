# from sklearn.datasets import fetch_20newsgroups
# newsgroups_train = fetch_20newsgroups(subset='train')
from sklearn.feature_extraction.text import TfidfVectorizer
count_vect = TfidfVectorizer()
text = [u"today is a good day more bad day",u"tomorrow can be a bad day to bad one ", u"bad bad day",u"today today today"]
text_test = [u"test good day",u"more day to test"]
X_train_counts = count_vect.fit(text) #get the count of all words 
X_test_counts = count_vect.fit_transform(text_test) #get the count of all words 

from pprint import pprint
# print X_train_counts.toarray()
print X_test_counts.toarray() # only print out the non-zero cell

# pprint((X_train_counts.todense()))#get all counts
# pprint(count_vect.vocabulary_)#get all words in all docs 