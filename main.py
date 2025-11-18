import pandas as pd
import pickle
from preprocess import preprocess_text   # ✅ new spaCy-based preprocessing

# ---------------------------------------------------
# LOAD YOUR DATA
# ---------------------------------------------------
df = pd.read_csv('data/papers.csv')
df = df.iloc[:5000, :]      # keep first 5000 for performance

df['paper_text'] = df['paper_text'].fillna("")

# ---------------------------------------------------
# APPLY SPAcy PREPROCESSING
# ---------------------------------------------------
print("Preprocessing documents using spaCy...")

docs = df['paper_text'].apply(lambda x: preprocess_text(x))

print(" Preprocessing complete.")


# ---------------------------------------------------
# TRAIN TF-IDF MODEL
# ---------------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

print("Training CountVectorizer + TF-IDF...")

cv = CountVectorizer(
    max_df=0.95,
    max_features=8000,       # higher for better vocabulary with lemmas
    ngram_range=(1, 3)       # 1–3 grams still useful
)

word_count_vector = cv.fit_transform(docs)

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vector)

feature_names = cv.get_feature_names_out()

print("Model training complete.")


# ---------------------------------------------------
# OPTIONAL: Extract keywords per row (topN = 10)
# ---------------------------------------------------
def get_keywords(idx, docs, topN=10):

    # Transform document to count vector then TF-IDF
    count_vec = cv.transform([docs[idx]])
    tfidf_vec = tfidf_transformer.transform(count_vec)

    tfidf_vec = tfidf_vec.tocoo()
    tuples = zip(tfidf_vec.col, tfidf_vec.data)

    sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    sorted_items = sorted_items[:topN]

    keywords = {feature_names[i]: round(float(score), 3)
                for i, score in sorted_items}

    return keywords


print("🔄 Extracting sample keywords for verification...")
sample_keywords = get_keywords(0, docs)
print("Example Keywords (Doc 0):")
print(sample_keywords)


# ---------------------------------------------------
# SAVE MODELS & FEATURES
# ---------------------------------------------------
print("Saving model files...")

pickle.dump(cv, open("models/count_vectorizer.pkl", "wb"))
pickle.dump(tfidf_transformer, open("models/tfidf_transformer.pkl", "wb"))
pickle.dump(feature_names, open("models/features_names.pkl", "wb"))

print("All done!")
print("Models saved in: models/")
