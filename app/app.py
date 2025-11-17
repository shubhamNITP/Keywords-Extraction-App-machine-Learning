from flask import Flask, request, render_template, jsonify
import pickle
import os
from preprocess import preprocess_text

app = Flask(__name__)

# --- Load Model Files Dynamically ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

cv = pickle.load(open(os.path.join(MODEL_DIR, "count_vectorizer.pkl"), "rb"))
tfidf_transformer = pickle.load(open(os.path.join(MODEL_DIR, "tfidf_transformer.pkl"), "rb"))
features_names = pickle.load(open(os.path.join(MODEL_DIR, "features_names.pkl"), "rb"))

# --- Keyword Extraction Function ---
def get_keywords(text, topN=10):
    text = preprocess_text(text)
    response = {}
    docs_words_count = tfidf_transformer.transform(cv.transform([text])).tocoo()
    tuples = zip(docs_words_count.col, docs_words_count.data)
    sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)[:topN]
    for idx, score in sorted_items:
        response[features_names[idx]] = round(score, 3)
    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract():
    user_text = request.form['text']
    keywords = get_keywords(user_text)
    return render_template('index.html', keywords=keywords, text=user_text)

# Optional JSON API for JavaScript integration
@app.route('/api/extract', methods=['POST'])
def api_extract():
    data = request.get_json()
    text = data.get("text", "")
    keywords = get_keywords(text)
    return jsonify(keywords)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
