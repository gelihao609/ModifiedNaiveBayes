from itertools import islice

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

from StemmingHelper import StemmingHelper
# print StemmingHelper.stem('learning')
import re
file = open('output.txt', 'r')
# .lower() returns a version with all upper case characters replaced with lower case characters.
text = file.read().lower()
file.close()
# replaces anything that is not a lowercase letter, a space, or an apostrophe with a space:
text = re.sub('[^a-z\ \']+', " ", text)

cleanWords = []
words = list(text.split())

from stop_words import get_stop_words
stop_words = get_stop_words('english')
for word in words:
	if word in stop_words: 
		words.remove(word)
chunked = list(chunk(words,20))
print chunked

from gensim.models import Word2Vec
min_count = 2
size = 50
window = 8

model = Word2Vec(chunked, min_count=min_count, size=size, window=window)
print model.most_similar('machine')