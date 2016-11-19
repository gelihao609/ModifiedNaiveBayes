from nltk.stem import LancasterStemmer, SnowballStemmer, PorterStemmer, WordNetLemmatizer
 
# stemmer = PorterStemmer()
stemmer = LancasterStemmer()
# stemmer = SnowballStemmer("english")

lemmatiser = WordNetLemmatizer()
 
print("Stem %s: %s" % ("studying", stemmer.stem("studying studied studies")))
print("Stem %s: %s" % ("studying", stemmer.stem("studied")))
print("Stem %s: %s" % ("studying", stemmer.stem("studies")))

# print("Lemmatise %s: %s" % ("studying", lemmatiser.lemmatize("studying")))
# print("Lemmatise %s: %s" % ("studying", lemmatiser.lemmatize("studied")))
# print("Lemmatise %s: %s" % ("studying", lemmatiser.lemmatize("studies")))

# print("Lemmatise %s: %s" % ("studying", lemmatiser.lemmatize("studying", pos="v")))