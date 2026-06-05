
import pickle
from preprocessing import clean_text

def predict_fake_news(text, model_name='logistic_regression'):
    """
    Predict if a news article is fake or real
    Compatible with Streamlit interface
    """
    # Load model and vectorizer
    model_filename = f'model_{model_name}.pkl'

    try:
        with open(model_filename, 'rb') as f:
            model = pickle.load(f)

        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model files not found. Please ensure {model_filename} and tfidf_vectorizer.pkl exist.")

    # Clean and transform text
    cleaned_text = clean_text(text)
    text_vector = vectorizer.transform([cleaned_text])

    # Make prediction
    prediction = model.predict(text_vector)[0]

    # Get probability if available
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(text_vector)[0]
        fake_prob = prob[0] if model.classes_[0] == 'FAKE' else prob[1]
        real_prob = prob[1] if model.classes_[1] == 'REAL' else prob[0]

        return {
            'prediction': prediction,
            'confidence': max(fake_prob, real_prob),
            'fake_probability': fake_prob,
            'real_probability': real_prob,
            'cleaned_text': cleaned_text
        }
    else:
        return {
            'prediction': prediction,
            'cleaned_text': cleaned_text
        }
