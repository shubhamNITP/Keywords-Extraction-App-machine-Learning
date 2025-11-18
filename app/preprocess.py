import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

# Add custom stopwords
CUSTOM_STOP = {
    "fig", "figure", "image", "sample", "using", "show", "result", "large",
    "also", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "table", "et", "al"
}

STOPWORDS = STOP_WORDS.union(CUSTOM_STOP)


# ---------------------------------------------------------
# PDF CLEANING
# ---------------------------------------------------------
def clean_pdf_text(text):

    # Remove reference numbers like [1], [12], (3)
    text = re.sub(r"\[\d+\]", " ", text)
    text = re.sub(r"\(\d+\)", " ", text)

    # Fix hyphenated words split across lines: comput- er → computer
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

    # Replace newlines with space
    text = text.replace("\n", " ")

    # Remove weird unicode chars
    text = text.encode("ascii", "ignore").decode()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------
# MAIN PREPROCESS FUNCTION — spaCy lemmatization
# ---------------------------------------------------------
def preprocess_text(text):

    # Step 1: Clean PDF junk
    text = clean_pdf_text(text)

    # Step 2: spaCy NLP pipeline
    doc = nlp(text.lower())

    tokens = []
    for token in doc:

        # Remove stopwords, punctuation, numbers
        if token.is_stop or token.is_punct or token.like_num:
            continue

        lemma = token.lemma_.strip()

        # Remove short terms & custom stopwords
        if len(lemma) < 4:
            continue
        if lemma in STOPWORDS:
            continue

        tokens.append(lemma)

    return " ".join(tokens)
