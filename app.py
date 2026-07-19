import io
import time
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pickle
import re
import string

import requests
import streamlit as st
from bs4 import BeautifulSoup

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# =============================================================================
# NLTK setup — only downloads what's missing, and stays quiet about it
# =============================================================================
def _ensure_nltk_data():
    lookups = [
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
    ]
    for pkg, path in lookups:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_nltk_data()

# =============================================================================
# Page config & constants
# =============================================================================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# NOTE: these are the actual filenames present in the repo. The previous
# version of app.py loaded "model.pkl" / "vectorizer.pkl", which don't
# exist — that's fixed here.
MODEL_FILES = {
    "Logistic Regression": "model_logistic_regression.pkl",
    "Random Forest": "model_random_forest.pkl",
    "SVM": "model_svm.pkl",
}
VECTORIZER_FILE = "tfidf_vectorizer.pkl"

# Fill these in with your real numbers from Project_Exhibition.ipynb —
# left as None so the app never shows a made-up accuracy figure.
MODEL_ACCURACY = {
    "Logistic Regression": 97.95,
    "Random Forest": 95.38,
    "SVM": 99.09,
}

SAMPLE_ARTICLE = (
    "Scientists have developed a new method for detecting early signs of "
    "extreme weather events using satellite imagery and machine learning "
    "algorithms. The research, published in a peer-reviewed journal, shows "
    "a significant improvement in prediction accuracy compared to traditional "
    "forecasting methods. Researchers say the technique could help communities "
    "prepare earlier for hurricanes, floods, and wildfires by giving local "
    "authorities several additional days of warning time."
)

LOADING_STEPS_TEXT = [
    ("🧹", "Cleaning text..."),
    ("🔢", "Vectorizing with TF-IDF..."),
    ("🤖", "Running model..."),
    ("📊", "Predicting..."),
]
LOADING_STEPS_URL = [
    ("🌐", "Scraping article..."),
] + LOADING_STEPS_TEXT

# =============================================================================
# Theme (light / dark toggle for the custom components only — Streamlit's
# own chrome still follows the user's system/app theme setting)
# =============================================================================
PALETTES = {
    "Light": dict(bg="#F5F7FA", card="#FFFFFF", text="#1F2430", subtext="#5B6472",
                  border="#E3E7ED", green="#2E7D32", red="#C62828", blue="#1565C0"),
    "Dark": dict(bg="#12151C", card="#1B1F2A", text="#EAEDF2", subtext="#9AA3B2",
                 border="#2B3140", green="#4CAF50", red="#EF5350", blue="#42A5F5"),
}


def inject_css(palette):
    st.markdown(
        f"""
        <style>
        .fnd-hero {{
            text-align: center;
            padding: 8px 0 4px 0;
        }}
        .fnd-hero h1 {{
            font-size: 2.4rem;
            margin-bottom: 4px;
            color: {palette['blue']};
        }}
        .fnd-hero p {{
            color: {palette['subtext']};
            font-size: 1.05rem;
            margin-top: 0;
        }}
        .fnd-card {{
            background: {palette['card']};
            border: 1px solid {palette['border']};
            border-radius: 14px;
            padding: 22px 26px;
            margin-bottom: 18px;
        }}
        .fnd-stats {{
            display: flex;
            gap: 14px;
            flex-wrap: wrap;
            justify-content: center;
            margin-bottom: 10px;
        }}
        .fnd-stat-pill {{
            background: {palette['card']};
            border: 1px solid {palette['border']};
            border-radius: 999px;
            padding: 8px 18px;
            font-size: 0.9rem;
            color: {palette['text']};
        }}
        .fnd-result-title {{
            font-size: 1.6rem;
            font-weight: 700;
            margin-bottom: 4px;
        }}
        .fnd-real {{ color: {palette['green']}; }}
        .fnd-fake {{ color: {palette['red']}; }}
        .fnd-meta-row {{
            display: flex;
            gap: 28px;
            flex-wrap: wrap;
            margin-top: 14px;
        }}
        .fnd-meta-item {{
            color: {palette['text']};
        }}
        .fnd-meta-item span {{
            display: block;
            font-size: 0.78rem;
            color: {palette['subtext']};
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .fnd-bar-label {{
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            color: {palette['text']};
            margin-bottom: 3px;
        }}
        .fnd-bar-track {{
            width: 100%;
            height: 12px;
            border-radius: 6px;
            background: {palette['border']};
            overflow: hidden;
            margin-bottom: 12px;
        }}
        .fnd-bar-fill {{
            height: 100%;
            border-radius: 6px;
        }}
        .fnd-gauge {{
            width: 130px;
            height: 130px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3rem;
            font-weight: 700;
            color: {palette['text']};
            margin: 0 auto 12px auto;
        }}
        .fnd-footer {{
            text-align: center;
            color: {palette['subtext']};
            font-size: 0.85rem;
            padding-top: 18px;
        }}
        .fnd-chip {{
            display: inline-block;
            background: {palette['border']};
            border-radius: 999px;
            padding: 3px 12px;
            margin: 3px;
            font-size: 0.8rem;
            color: {palette['text']};
        }}
        .stButton>button {{
            border-radius: 10px;
            font-weight: 600;
            padding: 0.6em 1.2em;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Cached loaders
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_vectorizer():
    with open(VECTORIZER_FILE, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_model(model_key):
    with open(MODEL_FILES[model_key], "rb") as f:
        return pickle.load(f)


# =============================================================================
# NLP helpers
# =============================================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[\d]+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"<.*?>", "", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)


def extract_article_from_url(url):
    """Scrape article text + best-effort metadata from a news URL."""
    try:
        resp = requests.get(
            url, timeout=8, headers={"User-Agent": "Mozilla/5.0 (compatible; FakeNewsDetector/1.0)"}
        )
        soup = BeautifulSoup(resp.text, "html.parser")

        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs).strip()
        if not text:
            return None

        def meta(*names):
            for name in names:
                tag = soup.find("meta", attrs={"property": name}) or soup.find(
                    "meta", attrs={"name": name}
                )
                if tag and tag.get("content"):
                    return tag["content"]
            return None

        title = None
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        title = meta("og:title") or title

        source = meta("og:site_name")
        if not source:
            try:
                source = url.split("//")[-1].split("/")[0].replace("www.", "")
            except Exception:
                source = None

        author = meta("author", "article:author")
        published = meta("article:published_time", "date", "pubdate")

        return {
            "text": text,
            "title": title,
            "source": source,
            "author": author,
            "published": published,
        }
    except Exception:
        return None


def predict_news(model, vectorizer, raw_text):
    start = time.time()
    cleaned = clean_text(raw_text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0] if hasattr(model, "predict_proba") else None
    elapsed = time.time() - start

    class_probs = {}
    if probs is not None:
        for cls, p in zip(model.classes_, probs):
            class_probs[cls] = p * 100
    confidence = max(class_probs.values()) if class_probs else None

    return prediction, confidence, class_probs, elapsed, vec


def explain_prediction(model, vectorizer, vec, top_n=8):
    """Return top contributing words for linear models (coef_ based)."""
    if not hasattr(model, "coef_"):
        return None
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    pos_class, neg_class = model.classes_[1], model.classes_[0]

    nz_indices = vec.nonzero()[1]
    contributions = [
        (feature_names[i], vec[0, i] * coefs[i]) for i in nz_indices
    ]
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top = contributions[:top_n]
    return [
        {"word": w, "score": s, "pushes_toward": pos_class if s > 0 else neg_class}
        for w, s in top
    ]


def run_loading_animation(steps, container):
    for icon, label in steps:
        container.markdown(f"{icon} {label}")
        time.sleep(0.28)
    container.empty()


# =============================================================================
# Session state
# =============================================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "text_input_value" not in st.session_state:
    st.session_state.text_input_value = ""
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

palette = PALETTES[st.session_state.theme]
inject_css(palette)

# =============================================================================
# Sidebar — model choice, theme, stats, history
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    st.session_state.theme = st.radio(
        "Appearance", options=["Light", "Dark"],
        index=["Light", "Dark"].index(st.session_state.theme),
        horizontal=True,
    )

    model_choice = st.selectbox("Model", list(MODEL_FILES.keys()), index=0)

    st.markdown("---")
    st.markdown("### 🧠 Models Available")
    for name in MODEL_FILES:
        acc = MODEL_ACCURACY.get(name)
        acc_txt = f"{acc:.1f}%" if acc is not None else "—"
        st.markdown(f"✔ **{name}** · accuracy: {acc_txt}")

    st.markdown("---")
    st.markdown("### 🕘 Recent Predictions")
    if st.session_state.history:
        for item in reversed(st.session_state.history[-6:]):
            icon = "✅" if item["prediction"] == "REAL" else "❌"
            st.markdown(f"{icon} **{item['prediction']}** · {item['confidence']:.1f}% · {item['model']}")
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No predictions yet this session.")

# =============================================================================
# Hero section
# =============================================================================
st.markdown(
    """
    <div class="fnd-hero">
        <h1>📰 Fake News Detector</h1>
        <p>Detect misinformation using Machine Learning and Natural Language Processing.<br>
        Analyze article text or news URLs in seconds.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="fnd-stats">
        <div class="fnd-stat-pill">✔ TF-IDF</div>
        <div class="fnd-stat-pill">✔ Logistic Regression</div>
        <div class="fnd-stat-pill">✔ Random Forest</div>
        <div class="fnd-stat-pill">✔ SVM</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Load model / vectorizer
# =============================================================================
load_error = None
try:
    vectorizer = load_vectorizer()
    model = load_model(model_choice)
except FileNotFoundError as e:
    load_error = str(e)

if load_error:
    st.error(
        f"⚠ Couldn't load model files ({load_error}). Make sure "
        f"`{VECTORIZER_FILE}` and `{MODEL_FILES[model_choice]}` are in the "
        "same folder as this app."
    )
    st.stop()

# =============================================================================
# Top-level tabs: Detector vs Model Comparison
# =============================================================================
main_tab_detect, main_tab_compare = st.tabs(["🔍 Detector", "📊 Model Comparison"])

with main_tab_detect:
    # =============================================================================
    # Input card — tabs for Text / URL
    # =============================================================================
    st.markdown('<div class="fnd-card">', unsafe_allow_html=True)

tab_text, tab_url = st.tabs(["📝 Text", "🔗 URL"])
article_text = None
url_meta = None

with tab_text:
    col_a, col_b = st.columns([5, 1])
    with col_b:
        if st.button("🎲 Try Sample", use_container_width=True):
            st.session_state.text_input_value = SAMPLE_ARTICLE
            st.rerun()

    text_val = st.text_area(
        "Paste Article",
        value=st.session_state.text_input_value,
        height=220,
        placeholder='Paste the complete news article here...\n\nExample:\n"Scientists have developed..."',
        key="text_area_widget",
    )
    st.session_state.text_input_value = text_val

    analyze_text_clicked = st.button("🔍 Analyze Article", type="primary", use_container_width=True, key="analyze_text")

with tab_url:
    url_val = st.text_input(
        "Article URL",
        placeholder="https://www.bbc.com/news/...",
        key="url_input_widget",
    )
    analyze_url_clicked = st.button("🔍 Analyze Article", type="primary", use_container_width=True, key="analyze_url")

st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# Handle analysis
# =============================================================================
run_analysis = False
input_source = None

if analyze_text_clicked:
    if not st.session_state.text_input_value.strip():
        st.warning("⚠ Please enter some text.")
    else:
        article_text = st.session_state.text_input_value
        run_analysis = True
        input_source = "text"

if analyze_url_clicked:
    if not url_val.strip():
        st.warning("⚠ Please enter a URL.")
    else:
        status_box = st.empty()
        status_box.markdown("🌐 Scraping article...")
        url_meta = extract_article_from_url(url_val)
        status_box.empty()
        if not url_meta:
            st.error("❌ Could not extract text from the provided URL.")
        else:
            article_text = url_meta["text"]
            run_analysis = True
            input_source = "url"

if run_analysis and article_text:
    loading_box = st.empty()
    steps = LOADING_STEPS_URL if input_source == "url" else LOADING_STEPS_TEXT
    run_loading_animation(steps, loading_box)

    prediction, confidence, class_probs, elapsed, vec = predict_news(model, vectorizer, article_text)

    st.session_state.history.append(
        {
            "prediction": prediction,
            "confidence": confidence,
            "model": model_choice,
            "time": datetime.now().strftime("%H:%M:%S"),
        }
    )

    is_real = prediction == "REAL"
    result_class = "fnd-real" if is_real else "fnd-fake"
    result_icon = "✅" if is_real else "❌"
    result_label = "REAL NEWS" if is_real else "FAKE NEWS"
    bar_color = palette["green"] if is_real else palette["red"]

    st.markdown('<div class="fnd-card">', unsafe_allow_html=True)

    col_gauge, col_details = st.columns([1, 2])

    with col_gauge:
        gauge_deg = int(round((confidence or 0) * 3.6))
        st.markdown(
            f"""
            <div class="fnd-gauge" style="background: conic-gradient({bar_color} {gauge_deg}deg, {palette['border']} 0deg);">
                <div style="background:{palette['card']}; width:100px; height:100px; border-radius:50%;
                            display:flex; align-items:center; justify-content:center; flex-direction:column;">
                    <div style="font-size:1.3rem;">{confidence:.1f}%</div>
                    <div style="font-size:0.7rem; color:{palette['subtext']};">confidence</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_details:
        st.markdown(f'<div class="fnd-result-title {result_class}">{result_icon} {result_label}</div>', unsafe_allow_html=True)

        for cls, pct in sorted(class_probs.items(), key=lambda x: -x[1]):
            cls_color = palette["green"] if cls == "REAL" else palette["red"]
            st.markdown(
                f"""
                <div class="fnd-bar-label"><span>{cls}</span><span>{pct:.1f}%</span></div>
                <div class="fnd-bar-track">
                    <div class="fnd-bar-fill" style="width:{pct:.1f}%; background:{cls_color};"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="fnd-meta-row">
                <div class="fnd-meta-item"><span>Model</span>{model_choice}</div>
                <div class="fnd-meta-item"><span>Processing Time</span>{elapsed:.2f}s</div>
                <div class="fnd-meta-item"><span>Risk</span>{"Low" if is_real else "High"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Article metadata (URL mode only)
    if url_meta:
        st.markdown("---")
        st.markdown("**📄 Article Metadata**")
        meta_cols = st.columns(4)
        meta_cols[0].markdown(f"**Title**\n\n{url_meta['title'] or '—'}")
        meta_cols[1].markdown(f"**Source**\n\n{url_meta['source'] or '—'}")
        meta_cols[2].markdown(f"**Author**\n\n{url_meta['author'] or '—'}")
        meta_cols[3].markdown(f"**Published**\n\n{url_meta['published'] or '—'}")

    # Explainability
    explanation = explain_prediction(model, vectorizer, vec)
    if explanation:
        st.markdown("---")
        st.markdown("**🔎 What influenced this prediction**")
        chips = ""
        for item in explanation:
            color = palette["green"] if item["pushes_toward"] == "REAL" else palette["red"]
            chips += f'<span class="fnd-chip" style="border:1px solid {color}; color:{color};">{item["word"]}</span>'
        st.markdown(chips, unsafe_allow_html=True)
    elif model_choice == "Random Forest":
        st.caption("ℹ Word-level explanations are shown for linear models (Logistic Regression, SVM). Random Forest doesn't expose per-word contributions the same way.")

    # Export
    report = (
        f"Fake News Detector — Result\n"
        f"Prediction: {result_label}\n"
        f"Confidence: {confidence:.1f}%\n"
        f"Model: {model_choice}\n"
        f"Processing Time: {elapsed:.2f}s\n"
        f"Generated: {datetime.now().isoformat(timespec='seconds')}\n"
    )
    st.download_button(
        "⬇ Download Result",
        data=report,
        file_name="fake_news_result.txt",
        mime="text/plain",
    )

    st.markdown("</div>", unsafe_allow_html=True)

with main_tab_compare:
    st.markdown('<div class="fnd-card">', unsafe_allow_html=True)
    st.subheader("Model Performance Comparison")

    comparison_data = {
        "Model": ["Logistic Regression", "Random Forest", "SVM"],
       "Accuracy": [0.9861, 0.9937, 0.9932],
        "Precision": [0.9906, 0.9953, 0.9947],
        "Recall": [0.9828, 0.9925, 0.9923],
        "F1-Score": [0.9866, 0.9939, 0.9935],
        "Training Time (s)": [1.80, 3.58, 4.22],
    }
    df_comparison = pd.DataFrame(comparison_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(comparison_data["Model"]))
    width = 0.2
    colors = ["#1565C0", "#C62828", "#2E7D32", "#F39C12"]

    for i, metric in enumerate(metrics):
        values = df_comparison[metric].values
        ax1.bar(x + i * width, values, width, label=metric, color=colors[i], alpha=0.85)

    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(df_comparison["Model"])
    ax1.set_ylim(0.85, 1.02)
    ax1.set_title("Performance Metrics")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    times = df_comparison["Training Time (s)"].values
    ax2.bar(df_comparison["Model"], times, color=["#1565C0", "#2E7D32", "#C62828"], alpha=0.8)
    ax2.set_yscale("log")
    ax2.set_title("Training Time (log scale, seconds)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(df_comparison, use_container_width=True, hide_index=True)

    st.markdown(
        """
        **Recommendation:** Logistic Regression offers the best balance —
        strong accuracy (97.95%) with training time under 2 seconds.
        SVM edges out slightly higher accuracy (99.09%) but takes over
        2 hours to train, making it impractical to retrain often.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# Footer
# =============================================================================
st.markdown(
    """
    <div class="fnd-footer">
        Built with 🐍 Python · Scikit-Learn · NLTK · BeautifulSoup · Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)