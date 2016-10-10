# from sklearn.datasets import fetch_20newsgroups
# newsgroups_train = fetch_20newsgroups(subset='train')
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
train_data = [u"today is a good day more bad day",u"tomorrow can be a bad day to bad one ", u"bad bad day",u"today today today"]
train_target = [1,1,1,0]
print type(text[0])
X_train_counts = count_vect.fit_transform(text) #get the count of all words 
from pprint import pprint
# print X_train_counts # only print out the non-zero cell
# pprint((X_train_counts.todense()))#get all counts
# pprint(count_vect.vocabulary_)#get all words in all docs 

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(text)

# print vectorizer.get_feature_names()
pprint(type(vectors.toarray()))
# from sklearn.feature_extraction.text import TfidfVectorizer
# corpus = ["This is very strange",
#           "This is very nice"]
# vectorizer = TfidfVectorizer(min_df=1)
# X = vectorizer.fit_transform(corpus)
# idf = vectorizer.idf_
# print dict(zip(vectorizer.get_feature_names(), idf))

# vectors = tf_transformer1.fit_transform(text)
# tf_transformer = TfidfTransformer(norm='l2',use_idf=False).fit(X_train_counts)
# tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)

# X_train_tf = tf_transformer.transform(X_train_counts)

# pprint(X_train_tf.todense())#get all frequencies
# pprint(vectors.todense())#get all frequencies


# pprint(type(newsgroups_train.filenames))

###########################syntax###############
# vectors = vectorizer.fit_transform(text) --> sparse matrix

