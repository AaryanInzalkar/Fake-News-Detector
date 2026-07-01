from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

fake_df = pd.read_csv('Fake.csv')
fake_df['label'] = 'FAKE'


true_df = pd.read_csv('True.csv')
true_df['label'] = 'REAL'

df = pd.concat([fake_df, true_df], ignore_index=True)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(df.head())


def clean_text(text):

    text = text.lower()

    text = re.sub(r'[\d]+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    text = re.sub(r'<.*?>', '', text)

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)


df['cleaned_text'] = df['text'].apply(clean_text)

tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X = tfidf.fit_transform(df['cleaned_text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, pos_label='FAKE'))
print("Recall:", recall_score(y_test, y_pred, pos_label='FAKE'))
print("F1 Score:", f1_score(y_test, y_pred, pos_label='FAKE'))
