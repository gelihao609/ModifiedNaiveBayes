from nltk.corpus import stopwords
cachedStopWords = stopwords.words("english")

print cachedStopWords

def single_removeSW(article):
	return ' '.join([word for word in article.split(' ') if word not in cachedStopWords])

def removeSW(articles):
	noSW = []
	for num in articles:
		noSW.append(single_removeSW(num))
	return noSW