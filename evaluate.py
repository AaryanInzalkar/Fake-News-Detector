import pandas as pd
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from preprocessing import clean_text

print("Loading data...")
fake_df = pd.read_csv('Fake.csv')
fake_df['label'] = 'FAKE'
true_df = pd.read_csv('True.csv')
true_df['label'] = 'REAL'

df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Cleaning text (this mirrors main.py exactly)...")
t0 = time.time()
df['cleaned_text'] = df['text'].apply(clean_text)
print(f"  done in {time.time()-t0:.1f}s")

# Load the already-fit vectorizer instead of re-fitting, so we use
# EXACTLY the same feature space the saved models were trained on.
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

X = tfidf.transform(df['cleaned_text'])
y = df['label']

# Same split call, same random_state -> identical test set as training time
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Test set size: {X_test.shape[0]}")

model_files = {
    'Logistic Regression': 'model_logistic_regression.pkl',
    'Random Forest': 'model_random_forest.pkl',
    'SVM': 'model_svm.pkl',
}

print("\n" + "=" * 70)
print(f"{'Model':<22}{'Accuracy':<10}{'Precision':<11}{'Recall':<10}{'F1':<8}")
print("=" * 70)

for name, path in model_files.items():
    with open(path, 'rb') as f:
        model = pickle.load(f)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='FAKE')
    rec = recall_score(y_test, y_pred, pos_label='FAKE')
    f1 = f1_score(y_test, y_pred, pos_label='FAKE')
    print(f"{name:<22}{acc:<10.4f}{prec:<11.4f}{rec:<10.4f}{f1:<8.4f}")
    print(f"  Confusion matrix (rows=actual, cols=pred, order FAKE/REAL):")
    print("  " + str(confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])).replace("\n", "\n  "))
    print()

print("=" * 70)
print("Done. Compare these to the numbers shown in app.py's Model Comparison tab.")