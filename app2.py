import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Configure page
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for mature, professional styling
st.markdown("""
<style>
/* Main styling */
.main {
    padding: 2rem;
    font-family: 'Arial', sans-serif;
}

.main-header {
    font-size: 2.5rem;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 300;
    border-bottom: 2px solid #ecf0f1;
    padding-bottom: 1rem;
}

/* Result cards */
.result-card {
    padding: 2rem;
    border-radius: 8px;
    margin: 1.5rem 0;
    border-left: 4px solid;
    background: #fafafa;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.fake-result {
    border-left-color: #e74c3c;
    background: linear-gradient(135deg, #fff5f5 0%, #fef2f2 100%);
}

.real-result {
    border-left-color: #27ae60;
    background: linear-gradient(135deg, #f0fff4 0%, #f7fffa 100%);
}

.prediction-text {
    font-size: 1.8rem;
    font-weight: 600;
    margin: 0 0 0.5rem 0;
    color: #2c3e50;
}

.confidence-text {
    font-size: 1.2rem;
    color: #7f8c8d;
    margin: 0;
}

/* Input styling */
.stTextArea > div > div > textarea {
    font-size: 14px;
    line-height: 1.6;
}

.stTextInput > div > div > input {
    font-size: 14px;
}

/* Button styling */
.stButton > button {
    background: #3498db;
    color: white;
    border: none;
    padding: 0.7rem 2rem;
    font-size: 1rem;
    font-weight: 500;
    border-radius: 4px;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background: #2980b9;
    box-shadow: 0 2px 8px rgba(52, 152, 219, 0.3);
}

/* Sidebar styling */
.sidebar .sidebar-content {
    background: #f8f9fa;
}

.sidebar-header {
    font-size: 1.3rem;
    color: #2c3e50;
    font-weight: 600;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #dee2e6;
}

/* Metrics styling */
.metric-container {
    background: white;
    padding: 1rem;
    border-radius: 6px;
    border: 1px solid #e9ecef;
    margin: 0.5rem 0;
}

/* Table styling */
.dataframe {
    font-size: 14px;
}

.dataframe th {
    background-color: #f8f9fa;
    color: #495057;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# Text preprocessing function
def clean_text(text):
    """Clean and preprocess text data"""
    text = text.lower()
    text = re.sub(r'[\d]+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'<.*?>', '', text)

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)


# Prediction function
def predict_fake_news(text, model_name='logistic_regression'):
    """Predict if a news article is fake or real"""
    try:
        model_filename = f'model_{model_name}.pkl'

        with open(model_filename, 'rb') as f:
            model = pickle.load(f)

        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)

        cleaned_text = clean_text(text)
        text_vector = vectorizer.transform([cleaned_text])

        prediction = model.predict(text_vector)[0]

        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(text_vector)[0]
            fake_prob = prob[0] if model.classes_[0] == 'FAKE' else prob[1]
            real_prob = prob[1] if model.classes_[1] == 'REAL' else prob[0]
            confidence = max(fake_prob, real_prob)

            return {
                'prediction': prediction,
                'confidence': confidence
            }
        else:
            return {'prediction': prediction}

    except FileNotFoundError:
        st.error("Model files not found. Please ensure all model files are present.")
        return None
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None


# Web scraping function
def extract_article_from_url(url):
    """Extract article text from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()

        # Try to find article content using common selectors
        article_selectors = [
            'article', '.article-content', '.post-content', '.entry-content',
            '.content', '.article-body', '.story-body', '.article-text'
        ]

        article_text = ""
        for selector in article_selectors:
            elements = soup.select(selector)
            if elements:
                article_text = ' '.join([elem.get_text() for elem in elements])
                break

        # If no specific article content found, get paragraphs
        if not article_text:
            paragraphs = soup.find_all('p')
            article_text = ' '.join([p.get_text() for p in paragraphs])

        # Clean the text
        article_text = re.sub(r'\s+', ' ', article_text).strip()

        return article_text if len(article_text) > 100 else None

    except Exception as e:
        return None

# App header
st.markdown('<h1 class="main-header">Fake News Detection System</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown('<div class="sidebar-header">Model Settings</div>', unsafe_allow_html=True)

# Model selection
model_options = {
    'Logistic Regression (Recommended)': 'logistic_regression',
    'Random Forest': 'random_forest',
    'Support Vector Machine': 'svm'
}

selected_model_name = st.sidebar.selectbox(
    "Choose Model:",
    options=list(model_options.keys()),
    index=0
)

selected_model = model_options[selected_model_name]

# Display model performance info
model_info = {
    'logistic_regression': {
        'accuracy': '97.95%',
        'f1_score': '98.04%',
        'training_time': '1.65s'
    },
    'random_forest': {
        'accuracy': '95.38%',
        'f1_score': '95.67%',
        'training_time': '22.71s'
    },
    'svm': {
        'accuracy': '99.09%',
        'f1_score': '99.13%',
        'training_time': '2.2+ hours'
    }
}

st.sidebar.markdown("### Model Performance")
info = model_info[selected_model]
st.sidebar.markdown(f"**Accuracy:** {info['accuracy']}")
st.sidebar.markdown(f"**F1-Score:** {info['f1_score']}")
st.sidebar.markdown(f"**Training Time:** {info['training_time']}")

# Main content
tab1, tab2, tab3 = st.tabs(["Article Analysis", "Model Comparison", "About"])

with tab1:
    st.header("News Article Analysis")
    
    # Input method selection
    input_method = st.radio(
        "Select input method:",
        ["Enter text manually", "Extract from URL"],
        horizontal=True
    )
    
    # Initialize session state for extracted text
    if 'extracted_article_text' not in st.session_state:
        st.session_state.extracted_article_text = ""
    
    article_text = ""
    
    if input_method == "Enter text manually":
        article_text = st.text_area(
            "Enter news article text:",
            height=300,
            placeholder="Paste your news article here..."
        )
    
    else:
        article_url = st.text_input(
            "Enter article URL:",
            placeholder="https://example.com/news-article"
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Extract Article") and article_url:
                with st.spinner("Extracting article content..."):
                    extracted_text = extract_article_from_url(article_url)
                    
                    if extracted_text:
                        st.session_state.extracted_article_text = extracted_text
                        st.success("Article extracted successfully!")
                    else:
                        st.error("Failed to extract article content. Please check the URL or try entering text manually.")
        
        # Show extracted content if available
        if st.session_state.extracted_article_text:
            with st.expander("View extracted content", expanded=True):
                st.text_area("Extracted text:", value=st.session_state.extracted_article_text, height=200, disabled=True)
            
            # Set article_text to the extracted content
            article_text = st.session_state.extracted_article_text
    
    # Analysis section
    if st.button("Analyze Article", type="primary"):
        # Check both manual input and extracted text
        text_to_analyze = article_text or st.session_state.extracted_article_text
        
        if text_to_analyze.strip():
            with st.spinner("Analyzing article..."):
                result = predict_fake_news(text_to_analyze, selected_model)
                
                if result:
                    prediction = result['prediction']
                    confidence = result.get('confidence', 0)
                    
                    # Display result
                    if prediction == 'FAKE':
                        st.markdown(
                            f'''
                            <div class="result-card fake-result">
                                <div class="prediction-text">FAKE NEWS</div>
                                <div class="confidence-text">Confidence: {confidence:.1%}</div>
                            </div>
                            ''',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'''
                            <div class="result-card real-result">
                                <div class="prediction-text">REAL NEWS</div>
                                <div class="confidence-text">Confidence: {confidence:.1%}</div>
                            </div>
                            ''',
                            unsafe_allow_html=True
                        )
        else:
            st.warning("Please enter some text or extract from URL to analyze.")
    
    # Add a clear button for URL extraction mode
    if input_method == "Extract from URL" and st.session_state.extracted_article_text:
        if st.button("Clear Extracted Text"):
            st.session_state.extracted_article_text = ""
            st.rerun()

with tab2:
    st.header("Model Performance Comparison")
    
    # Performance data
    comparison_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'SVM'],
        'Accuracy': [0.9795, 0.9538, 0.9909],
        'Precision': [0.9819, 0.9414, 0.9923],
        'Recall': [0.9790, 0.9724, 0.9902],
        'F1-Score': [0.9804, 0.9567, 0.9913],
        'Training Time (seconds)': [1.65, 22.71, 8053.65]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Performance metrics visualization
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Performance metrics comparison
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(comparison_data['Model']))
        width = 0.2
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for i, metric in enumerate(metrics):
            values = df_comparison[metric].values
            bars = ax1.bar(x + i*width, values, width, label=metric, color=colors[i], alpha=0.8)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Performance Metrics Comparison')
        ax1.set_xticks(x + width*1.5)
        ax1.set_xticklabels(df_comparison['Model'])
        ax1.legend()
        ax1.set_ylim(0.85, 1.02)
        ax1.grid(True, alpha=0.3)
        
        # Training time comparison
        times = df_comparison['Training Time (seconds)'].values
        bars = ax2.bar(df_comparison['Model'], times, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.7)
        
        for bar, time_val in zip(bars, times):
            if time_val < 60:
                label = f'{time_val:.1f}s'
            elif time_val < 3600:
                label = f'{time_val/60:.1f}min'
            else:
                label = f'{time_val/3600:.1f}hr'
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                    label, ha='center', va='bottom')
        
        ax2.set_ylabel('Training Time (seconds)')
        ax2.set_title('Training Time Comparison')
        ax2.set_yscale('log')
        
        # Fix the log scale formatting to avoid mathtext parsing error
        from matplotlib.ticker import LogFormatterMathtext, LogLocator
        ax2.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        ax2.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
        
        # Manually set y-axis tick labels to avoid parsing issues
        ax2.set_yticks([1, 10, 100, 1000, 10000])
        ax2.set_yticklabels(['1', '10', '100', '1K', '10K'])
        
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Cross-validation results
        cv_data = [0.9811, 0.9631, 0.9920]
        cv_std = [0.0033, 0.0245, 0.0027]
        
        bars = ax3.bar(df_comparison['Model'], cv_data, yerr=cv_std, capsize=5,
                       color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.7)
        
        for bar, mean_val in zip(bars, cv_data):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{mean_val:.3f}', ha='center', va='bottom')
        
        ax3.set_ylabel('F1-Score')
        ax3.set_title('Cross-Validation Results (5-fold)')
        ax3.set_ylim(0.90, 1.0)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Efficiency vs Performance
        f1_scores = df_comparison['F1-Score'].values
        # Use regular scale instead of log to avoid mathtext issues
        times_normalized = df_comparison['Training Time (seconds)'].values
        
        scatter = ax4.scatter(times_normalized, f1_scores, c=['#3498db', '#2ecc71', '#e74c3c'], s=100, alpha=0.7)
        
        for i, model in enumerate(df_comparison['Model']):
            ax4.annotate(model, (times_normalized[i], f1_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax4.set_xlabel('Training Time (seconds)')
        ax4.set_ylabel('F1-Score')
        ax4.set_title('Efficiency vs Performance')
        ax4.set_xscale('log')
        
        # Fix log scale formatting for x-axis
        ax4.set_xticks([1, 10, 100, 1000, 10000])
        ax4.set_xticklabels(['1s', '10s', '100s', '1K s', '10K s'])
        
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error creating matplotlib visualization: {str(e)}")
        
        # Fallback: Create simple bar charts without log scales
        st.subheader("Performance Comparison (Alternative View)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Simple performance metrics chart
            fig_simple, ax_simple = plt.subplots(figsize=(10, 6))
            
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            x = np.arange(len(comparison_data['Model']))
            width = 0.2
            
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
            
            for i, metric in enumerate(metrics):
                values = df_comparison[metric].values
                ax_simple.bar(x + i*width, values, width, label=metric, color=colors[i], alpha=0.8)
            
            ax_simple.set_xlabel('Models')
            ax_simple.set_ylabel('Score')
            ax_simple.set_title('Performance Metrics Comparison')
            ax_simple.set_xticks(x + width*1.5)
            ax_simple.set_xticklabels(df_comparison['Model'])
            ax_simple.legend()
            ax_simple.set_ylim(0.85, 1.02)
            ax_simple.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig_simple)
        
        with col2:
            # Simple training time chart (linear scale)
            fig_time, ax_time = plt.subplots(figsize=(8, 6))
            
            # Convert large times to minutes/hours for better display
            times = df_comparison['Training Time (seconds)'].values
            time_labels = []
            time_values_display = []
            
            for time_val in times:
                if time_val < 60:
                    time_labels.append(f'{time_val:.1f}s')
                    time_values_display.append(time_val)
                elif time_val < 3600:
                    time_labels.append(f'{time_val/60:.1f}min')
                    time_values_display.append(time_val/60)
                else:
                    time_labels.append(f'{time_val/3600:.1f}hr')
                    time_values_display.append(time_val/3600)
            
            bars = ax_time.bar(range(len(df_comparison['Model'])), time_values_display, 
                              color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.7)
            
            for bar, label in zip(bars, time_labels):
                ax_time.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(time_values_display)*0.01,
                            label, ha='center', va='bottom')
            
            ax_time.set_ylabel('Training Time')
            ax_time.set_title('Training Time Comparison')
            ax_time.set_xticks(range(len(df_comparison['Model'])))
            ax_time.set_xticklabels(df_comparison['Model'], rotation=45)
            ax_time.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig_time)
    
    # Performance table
    st.subheader("Detailed Performance Metrics")
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)
    
    # Key insights
    st.subheader("Key Insights")
    st.markdown("""
    **Performance Analysis:**
    - **Logistic Regression** provides the best balance of performance and efficiency
    - **SVM** achieves highest accuracy but requires excessive training time (4,870x slower)
    - **Random Forest** offers moderate performance with interpretability benefits
    
    **Recommendation:**
    For production deployment, **Logistic Regression** is the optimal choice due to:
    - Excellent performance (98.04% F1-score)
    - Fast training (1.65 seconds)
    - Consistent cross-validation results
    - Lower computational requirements
    """)

with tab3:
    st.header("About This System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Project Overview")
        st.markdown("""
        This system uses machine learning to classify news articles as FAKE or REAL 
        based on textual content analysis.
        
        **Key Features:**
        - Advanced text preprocessing with NLP techniques
        - TF-IDF feature extraction with bi-gram support
        - Multiple model comparison (Logistic Regression, Random Forest, SVM)
        - URL-based article extraction
        - Real-time prediction with confidence scores
        
        **Technical Implementation:**
        - Text cleaning: tokenization, lemmatization, stopword removal
        - Feature engineering: TF-IDF vectorization
        - Model evaluation: cross-validation, multiple metrics
        - Web scraping: BeautifulSoup for article extraction
        """)
    
    with col2:
        st.subheader("Model Performance Summary")
        st.markdown("""
        **Logistic Regression (Recommended)**
        - Accuracy: 97.95%
        - F1-Score: 98.04%
        - Training Time: 1.65 seconds
        - Cross-validation: 98.11% ± 0.33%
        
        **Random Forest**
        - Accuracy: 95.38%
        - F1-Score: 95.67%
        - Training Time: 22.71 seconds
        - Cross-validation: 96.31% ± 2.45%
        
        **Support Vector Machine**
        - Accuracy: 99.09%
        - F1-Score: 99.13%
        - Training Time: 8,053.65 seconds
        - Cross-validation: 99.20% ± 0.27%
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #7f8c8d; padding: 1rem;'>"
    "<p>Fake News Detection System | Machine Learning Classification Tool</p>"
    "</div>", 
    unsafe_allow_html=True
)