# 🔍 Fake News Detector

A machine learning-powered web application that detects whether a news article is **real or fake** — using NLP preprocessing and multiple trained classification models.

---

## 📌 About the Project

With the rise of misinformation online, this tool helps users verify the credibility of news articles. Simply paste article text or enter a URL, and the model will classify it as **REAL** or **FAKE** along with a confidence score.

---

## 🚀 Features

- 📝 Paste raw article text for instant analysis
- 🔗 Enter a news article URL — the app scrapes and analyzes it automatically
- 📊 Confidence score displayed with every prediction
- 🧠 Multiple ML models trained and compared (Logistic Regression, Random Forest, SVM)
- 🌐 Clean Streamlit web interface

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Web Framework | Streamlit |
| ML Models | Logistic Regression, Random Forest, SVM |
| NLP | NLTK (tokenization, stopword removal, lemmatization) |
| Vectorization | TF-IDF |
| Web Scraping | BeautifulSoup, Requests |
| Data | Fake.csv, True.csv (news datasets) |

---

## 📁 Project Structure

```
Fake-News-Detector/
│
├── app.py                  # Main Streamlit app
├── app2.py                 # Alternate app version
├── main.py                 # Model training script
├── main2.py                # Alternate training script
├── preprocessing.py        # Text cleaning & preprocessing
├── prediction.py           # Prediction logic
├── Prototype.py            # Early prototype
│
├── model_logistic_regression.pkl   # Trained LR model
├── model_random_forest.pkl         # Trained RF model
├── model_svm.pkl                   # Trained SVM model
├── tfidf_vectorizer.pkl            # Saved TF-IDF vectorizer
│
├── Fake.csv                # Fake news dataset
├── True.csv                # Real news dataset
├── Figure_1.png            # Model performance graph
├── Flowchart.mmd           # Project flowchart
│
├── Project_Exhibition.ipynb        # Notebook for exhibition
├── Model Architecture.docx         # Model architecture doc
└── Automated Fake News Detection Using Machine Learning.docx
```

---

## ⚙️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/AaryanInzalkar/Fake-News-Detector.git
cd Fake-News-Detector
```

**2. Install dependencies**
```bash
pip install streamlit nltk scikit-learn beautifulsoup4 requests numpy
```

**3. Download NLTK data**
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

**4. Run the app**
```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. **Input** — User pastes article text or provides a URL
2. **Scraping** (if URL) — BeautifulSoup extracts article paragraphs
3. **Preprocessing** — Text is lowercased, punctuation removed, stopwords filtered, and words lemmatized
4. **Vectorization** — TF-IDF converts cleaned text to numerical features
5. **Prediction** — Trained model classifies as REAL or FAKE with a confidence score
6. **Output** — Result displayed on the Streamlit UI

---

## 📊 Models & Performance

Three models were trained and evaluated on the dataset:

- **Logistic Regression** — Fast, interpretable baseline
- **Random Forest** — Ensemble method for better generalization
- **SVM** — High-accuracy classifier for text classification tasks

---

## 👤 Author

- Aaryan Inzalkar
---

## 📄 License

This project is for academic purposes.
