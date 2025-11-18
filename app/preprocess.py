import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import os

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Setup NLTK paths
nltk.data.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tokenizers'))
nltk.data.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'corpora'))

# Stopwords
stop_words = set(stopwords.words('english'))
new_words = [
    "fig", "figure", "image", "sample", "using", "show", "result", "large",
    "also", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "table", "et", "al"
]
stop_words = stop_words.union(new_words)

stemmer = PorterStemmer()


# ------------------------------------------------------
# CLEANING UTILITIES
# ------------------------------------------------------

def clean_pdf_text(text):
    """
    Fix common PDF text issues before preprocessing.
    """

    # Remove references like [1], [12], (3), etc.
    text = re.sub(r"\[\d+\]", " ", text)
    text = re.sub(r"\(\d+\)", " ", text)

    # Remove double hyphen line breaks ("comput-\ner" -> "computer")
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

    # Replace newlines with space
    text = text.replace("\n", " ")

    # Remove non-ASCII characters
    text = text.encode("ascii", "ignore").decode()

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ------------------------------------------------------
# MAIN PREPROCESS FUNCTION
# ------------------------------------------------------

def preprocess_text(text):
    """
    Clean + normalize + tokenize + remove stopwords + stem
    """

    text = clean_pdf_text(text)

    # Lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Keep only alphabets
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove short words + stopwords
    tokens = [w for w in tokens if w not in stop_words and len(w) > 3]

    # Stemming
    tokens = [stemmer.stem(w) for w in tokens]

    return " ".join(tokens)
