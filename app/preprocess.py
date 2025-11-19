import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# Ensure required nltk files exist
for resource in ["stopwords", "punkt", "punkt_tab"]:
    try:
        if resource == "stopwords":
            nltk.data.find("corpora/stopwords")
        elif resource == "punkt":
            nltk.data.find("tokenizers/punkt")
        elif resource == "punkt_tab":
            nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download(resource)

stop_words = set(stopwords.words('english'))
extra_words = [
    "fig","figure","image","sample","using","show",
    "result","large","also","one","two","three","four",
    "five","six","seven","eight","nine"
]
stop_words.update(extra_words)

stemmer = PorterStemmer()

def preprocessing_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 3]
    tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(tokens)
