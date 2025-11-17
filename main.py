import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle

# Load the CSV file into a DataFrame
df = pd.read_csv('data/papers.csv')

# Display the first few rows of the DataFrame
# print(df.head())

# Display the shape of the DataFrame
# print(df.shape)

df = df.iloc[:5000,:] # Limit to first 5000 rows for performance

# Preprocessing the data 

nltk.data.path.append(r"C:\Users\shubh\Documents\FifthSemester\MachineLearning\Project") # Add custom nltk data path 
for resource in ["stopwords", "punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource)



# Stop words are words which do not contain important significance to be used in NLP tasks.
stop_words = set(stopwords.words('english')) # Get the set of stop words from nltk
new_words = ["fig", "figure", "image" , "sample" , "using" , "show" , "result" , "large" , "also" , "one" , "two" , "three" , "four" , "five" , "six" , "seven" , "eight" , "nine" ] # custom stop words
stop_words = list(stop_words.union(new_words)) # Combine nltk stop words with custom stop words

# print(stop_words)

def preprocessing_text(text):
    text = text.lower() # Convert to lowercase
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text) # Remove special characters and numbers
    text = nltk.word_tokenize(text) # Tokenization

    text = [word for word in text if word not in stop_words] # Remove stop words
    text = [word for word in text if len(word) >3] # Remove short words

    stemming = PorterStemmer() # Stemming
    text = [stemming.stem(word) for word in text] # Apply stemming i.e, reducing words to their root form
    return ' '.join(text) # Join the list of words back into a single string

df['paper_text'] = df['paper_text'].fillna('') # Fill NaN values with empty strings
docs = df['paper_text'].apply(lambda x : preprocessing_text(x))

# print(docs[0])

# Vectorization using Bag of Words
cv = CountVectorizer(max_df=0.95, max_features=5000 , ngram_range=(1 , 3)) # Initialize CountVectorizer with max document frequency and max features
word_count_vector = cv.fit_transform(docs) # Fit and transform the documents to create the word count vector

tfidf_transformer = TfidfTransformer(smooth_idf=True , use_idf=True) # Initialize TfidfTransformer
tfidf_transformer = tfidf_transformer.fit(word_count_vector) # Fit the transformer to the word count vector

features_names = cv.get_feature_names_out() # Get the feature names from the CountVectorizer

def get_keywords(idx , docs  , topN = 10):
    # getting word count and importance
    docs_words_count = tfidf_transformer.transform(cv.transform([docs[idx]]))

    # sorting sparse matrix to get the important words
    docs_words_count = docs_words_count.tocoo()

    tuples = zip(docs_words_count.col , docs_words_count.data)
    
    sorted_items = sorted(tuples , key=lambda x: (x[1] , x[0]) , reverse=True)

    # getting top n keywords
    sorted_items = sorted_items[:topN]

    score_vals = []
    features_vals = []
    for idx, score in sorted_items :
        score_vals.append(round(score , 3))
        features_vals.append(features_names[idx])

    # final result
    results = {}
    for idx in range(len(features_vals)):
        results[features_vals[idx]] = score_vals[idx]

    return results


def print_keywords(idx, keywords , df):
    print("\n==================title=======================")
    print(df['title'][idx])
    print("\n==================abstract====================")
    print(df['abstract'][idx])
    print("\n==================keywords====================")
    for k in keywords:
        print(k ,keywords[k])

# Create a new column to store extracted keywords
all_keywords = []

for idx in range(len(df)):
    keywords = get_keywords(idx, docs)
    top_keywords = ', '.join(keywords.keys())  # Join top keyword names
    all_keywords.append(top_keywords)

df['extracted_keywords'] = all_keywords

# Save to a new CSV
df.to_csv('data/papers_with_keywords.csv', index=False)

print(" Keywords extracted and saved to 'data/papers_with_keywords.csv'")

# Save the model files

pickle.dump(cv, open("models/count_vectorizer.pkl", "wb"))
pickle.dump(tfidf_transformer, open("models/tfidf_transformer.pkl", "wb"))
pickle.dump(features_names, open("models/features_names.pkl", "wb"))