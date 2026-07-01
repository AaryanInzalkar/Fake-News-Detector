import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle
import matplotlib.pyplot as plt


fake_df = pd.read_csv('Fake.csv')
fake_df['label'] = 'FAKE'


true_df = pd.read_csv('True.csv')
true_df['label'] = 'REAL'  


df = pd.concat([fake_df, true_df], ignore_index=True)


df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.head()


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

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X = tfidf.fit_transform(df['cleaned_text'])
y = df['label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, pos_label='FAKE'))
print("Recall:", recall_score(y_test, y_pred, pos_label='FAKE'))
print("F1 Score:", f1_score(y_test, y_pred, pos_label='FAKE'))

from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report)

print(df_report)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)

disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()


with open("model1.pkl", "wb") as model_file:
    pickle.dump(clf, model_file)

with open("vectorizer1.pkl", "wb") as vec_file:
    pickle.dump(tfidf, vec_file)
