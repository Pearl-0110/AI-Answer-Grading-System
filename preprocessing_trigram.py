import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams

def preprocess_text(text):
    text = str(text).lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    trigrams = ngrams(tokens, 3)
    return [' '.join(gram) for gram in trigrams]
