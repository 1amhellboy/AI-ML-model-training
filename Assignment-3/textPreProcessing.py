import requests
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab') # Download punkt_tab resource

url = 'https://raw.githubusercontent.com/dscape/spell/master/test/resources/big.txt'
text = requests.get(url).text

text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # remove special chars, digits
text = re.sub(r'\s+', ' ', text)  # extra spaces

tokens = nltk.word_tokenize(text.lower())

stop_words = set(stopwords.words('english'))
tokens = [w for w in tokens if w not in stop_words]

stemmer = PorterStemmer()
tokens = [stemmer.stem(w) for w in tokens]

freq = Counter(tokens)
print(freq.most_common(20))