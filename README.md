# 📘 DocAI – PDF Keyword Extraction Web App

DocAI is a lightweight PDF & Text Intelligence Tool that extracts **high-value keywords** using a trained **TF-IDF model**.  
It supports:

✓ Multi-PDF Upload  
✓ Text Input  
✓ TF-IDF Keyword Extraction  
✓ PDF Text Extraction (PyMuPDF)  
✓ Custom Stopwords  
✓ REST API Support  
✓ Clean, Fast UI  
✓ Ready for Render Deployment  

---

# 🚀 Deploy on Render

Follow this exact structure:

```
MLProject/
 ├── app/
 │    ├── app.py
 │    ├── preprocess.py
 │    ├── templates/
 │    │      └── index.html
 │    └── static/
 │           ├── style.css
 │           └── script.js   (optional)
 ├── models/
 │    ├── count_vectorizer.pkl
 │    ├── tfidf_transformer.pkl
 │    └── features_names.pkl
 ├── data/
 │    └── papers.csv (used only during training)
 ├── main.py
 ├── requirements.txt
 ├── README.md
```

---

# ⚙️ Render Settings

### **Build command**
```
pip install -r requirements.txt
```

### **Start command**
```
gunicorn app.app:app
```

### Disk requirements  
Default is enough (your model files are small).

---

# 🧠 How DocAI Works

### 1️⃣ Preprocessing (NLTK)
- lowercasing  
- removing HTML  
- removing special characters  
- tokenizing  
- stopword removal  
- stemming using PorterStemmer  

### 2️⃣ Vectorization (scikit-learn TF-IDF)
- bag-of-words via CountVectorizer  
- TF-IDF transformation  
- n-gram range (1, 3)

### 3️⃣ Keyword Extraction  
Top-N keywords sorted by TF-IDF score.

### 4️⃣ PDF Extraction  
Using **PyMuPDF (`fitz`)**.

---

# 🌐 API Usage

### POST `/api/extract`

#### Request:
```json
{
  "text": "Deep learning improves accuracy.",
  "top_k": 3
}
```

#### Response:
```json
[
  { "word": "deep learn", "score": 0.843 },
  { "word": "improv accuraci", "score": 0.721 }
]
```

---

# ▶️ Local Development

### Install dependencies
```
pip install -r requirements.txt
```

### (First time only) Train the model
```
python main.py
```

### Run the server
```
python app/app.py
```

App runs at:
```
http://localhost:5000
```

---

# 📦 requirements.txt

```
Flask
gunicorn
pymupdf==1.24.4
pandas
numpy
scikit-learn
nltk
regex
```

---

# 🧑‍💻 Author

**Shubham Chaudhary**  
Machine Learning & Backend Developer
