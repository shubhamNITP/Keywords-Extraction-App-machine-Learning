import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import os

nltk.data.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tokenizers'))
nltk.data.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'corpora'))

nltk.data.path.append(r"C:\Users\shubh\Documents\FifthSemester\MachineLearning\Project")
stop_words = set(stopwords.words('english'))
new_words = ["fig","figure","image","sample","using","show","result","large","also",
             "one","two","three","four","five","six","seven","eight","nine"]
stop_words = list(stop_words.union(new_words))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 3]
    tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(tokens)
