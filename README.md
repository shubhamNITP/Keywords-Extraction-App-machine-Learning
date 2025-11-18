# DocAI – PDF Intelligence Web App

An advanced PDF & text processing web application that extracts:

✓ Keywords (TF-IDF + SVD)
✓ Summary
✓ Word Cloud
✓ Multi-PDF upload
✓ Page selection (1-3,5)
✓ PDF Viewer
✓ Dark Mode
✓ CSV Export

---

## 🚀 Deployment on Render

### 1. Upload this project to GitHub with this structure:
MLProject/
 ├── app/
 │    ├── app.py
 │    ├── preprocess.py
 │    ├── templates/index.html
 │    └── static/
 │         ├── style.css
 │         └── script.js
 ├── models/
 │    ├── count_vectorizer.pkl
 │    ├── tfidf_transformer.pkl
 │    └── svd_model.pkl
 ├── requirements.txt
 ├── Procfile
 ├── runtime.txt

### 2. Create a new Web Service on Render
- Environment: Python 3
- Build Command:
    pip install -r requirements.txt
- Start Command:
    gunicorn app.app:app

### 3. Set disk size if needed
Render FS is ephemeral, but model files load fine.

### 4. Deploy
Render will build & start the service automatically.

---

## 💡 Notes
- No NLTK needed (spaCy only).
- PDF processed with PyMuPDF for accuracy.
- Word Cloud uses Chart.js plugin.

