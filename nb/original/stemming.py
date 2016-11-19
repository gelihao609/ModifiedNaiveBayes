from nltk.stem import LancasterStemmer, SnowballStemmer, PorterStemmer, WordNetLemmatizer
# stemmer = LancasterStemmer()
stemmer = PorterStemmer()
# 1. remove newline, 2, strip leading and trailing whitespaces 3.split with space 4. stem 5. recombine to one article
def single_stem_nospace(article):
	return ' '.join([stemmer.stem(num) for num in article.replace('\n',' ').replace('\t',' ').strip().split(' ')])

def single_stem(article):
	return ' '.join([stemmer.stem(num) for num in article.strip().split(' ')])

def stem(articles):
	stemmed = []
	for num in articles:
		stemmed.append(single_stem_nospace(num))
	return stemmed