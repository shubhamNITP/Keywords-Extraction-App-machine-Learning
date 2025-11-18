import os
import io
import json
import re
import pickle
import fitz  # PyMuPDF for fast PDF text extraction
from flask import Flask, request, render_template, jsonify

# ------------------------------------------------------
# AUTO PATH RESOLUTION
# ------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))          # app/
BASE_DIR = os.path.dirname(CURRENT_DIR)                           # MLProject/
MODEL_DIR = os.path.join(BASE_DIR, "models")                      # MLProject/models/

# ------------------------------------------------------
# IMPORT PREPROCESSOR
# ------------------------------------------------------
from preprocess import preprocess_text


app = Flask(__name__)

# ------------------------------------------------------
# LOAD TRAINED MODEL FILES
# ------------------------------------------------------
cv = pickle.load(open(os.path.join(MODEL_DIR, "count_vectorizer.pkl"), "rb"))
tfidf_transformer = pickle.load(open(os.path.join(MODEL_DIR, "tfidf_transformer.pkl"), "rb"))
feature_names = pickle.load(open(os.path.join(MODEL_DIR, "features_names.pkl"), "rb"))


# ------------------------------------------------------
# FAST PDF TEXT EXTRACTION (PyMuPDF)
# ------------------------------------------------------
def extract_text_from_pdf(file_obj, page_ranges=None):
    doc = fitz.open(stream=file_obj.read(), filetype="pdf")
    num_pages = len(doc)

    # Determine which pages to extract
    pages = set()
    if page_ranges:
        for (a, b) in page_ranges:
            for p in range(max(1, a) - 1, min(num_pages, b)):  # convert to zero-based
                pages.add(p)
    else:
        pages = set(range(num_pages))

    text = ""
    for p in sorted(pages):
        text += doc[p].get_text("text") + "\n"

    return text


# ------------------------------------------------------
# KEYWORD EXTRACTION USING TRAINED TF-IDF MODEL
# ------------------------------------------------------
def extract_keywords_trained(raw_text, top_k=10):
    if not raw_text.strip():
        return []

    # 1. Preprocess the PDF/text exactly as training
    clean_text = preprocess_text(raw_text)

    # 2. Convert to vector using trained CountVectorizer
    word_count_vector = cv.transform([clean_text])

    # 3. Compute TF-IDF using trained transformer
    tfidf_scores = tfidf_transformer.transform(word_count_vector)
    scores = tfidf_scores.toarray().flatten()

    # 4. Get top K words
    top_indices = scores.argsort()[::-1][:top_k]

    keywords = []
    for idx in top_indices:
        keywords.append({
            "word": feature_names[idx],
            "score": float(round(scores[idx], 3))
        })

    return keywords


# ------------------------------------------------------
# SIMPLE FAST SUMMARY
# ------------------------------------------------------
sentence_splitter = re.compile(r'(?<=[.!?])\s+')

def split_sentences(text):
    return [s.strip() for s in sentence_splitter.split(text) if len(s.strip()) > 6]

def summarize_text(text, n_sentences=3):
    sentences = split_sentences(text)
    if not sentences:
        return []

    # Simple summary: pick longest sentences
    ranked = sorted(sentences, key=len, reverse=True)
    return ranked[:n_sentences]


# ------------------------------------------------------
# PAGE RANGE PARSER
# ------------------------------------------------------
def parse_page_ranges(range_str):
    range_str = (range_str or "").strip()
    if not range_str:
        return None

    ranges = []
    for part in range_str.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            ranges.append((int(a), int(b)))
        else:
            ranges.append((int(part), int(part)))

    return ranges


# ------------------------------------------------------
# ROUTES
# ------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/extract", methods=["POST"])
def extract():
    files = request.files.getlist("pdf_file")
    text_input = request.form.get("text_input", "").strip()

    page_ranges_json = request.form.get("page_ranges", "[]")
    page_ranges_list = json.loads(page_ranges_json)

    top_k = int(request.form.get("top_k", 10))
    summary_count = int(request.form.get("summary_sentences", 3))

    results = []

    # CASE 1: Text input only
    if text_input and not files:
        keywords = extract_keywords_trained(text_input, top_k)
        summary = summarize_text(text_input, summary_count)

        return jsonify({
            "results": [{
                "filename": "Manual Text",
                "keywords": keywords,
                "summary": summary,
                "snippet": text_input[:1500]
            }]
        })

    # CASE 2: PDFs uploaded
    for idx, pdf in enumerate(files):
        if pdf.filename == "":
            continue

        ranges = parse_page_ranges(page_ranges_list[idx]) if idx < len(page_ranges_list) else None

        pdf_bytes = pdf.read()
        pdf_text = extract_text_from_pdf(io.BytesIO(pdf_bytes), ranges)

        keywords = extract_keywords_trained(pdf_text, top_k)
        summary = summarize_text(pdf_text, summary_count)

        results.append({
            "filename": pdf.filename,
            "keywords": keywords,
            "summary": summary,
            "snippet": pdf_text[:1500] + "..." if len(pdf_text) > 1500 else pdf_text
        })

    return jsonify({"results": results})


# ------------------------------------------------------
# RUN SERVER
# ------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
