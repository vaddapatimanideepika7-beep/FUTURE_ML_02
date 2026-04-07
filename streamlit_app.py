import streamlit as st
import nltk
import re
import string
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# ---------- NLTK Setup ----------
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet')

# ---------- UI ----------
st.title("🎫 Ticket Classifier App")

user_input = st.text_area("Enter your support ticket:")

# ---------- Data ----------
data = {
    'text': [
        'Payment failed refund needed',
        'App crashes on startup',
        'Cannot login to account',
        'Order not delivered',
        'Need help with features'
    ],
    'category': ['billing', 'technical', 'account', 'shipping', 'general'],
    'priority': ['high', 'high', 'medium', 'high', 'low']
}

df = pd.DataFrame(data)

# ---------- Preprocessing ----------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

df['clean'] = df['text'].apply(preprocess)

# ---------- Model ----------
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['clean'])

cat_encoder = LabelEncoder()
pri_encoder = LabelEncoder()

y_cat = cat_encoder.fit_transform(df['category'])
y_pri = pri_encoder.fit_transform(df['priority'])

cat_model = LogisticRegression()
pri_model = LogisticRegression()

cat_model.fit(X, y_cat)
pri_model.fit(X, y_pri)

# ---------- Prediction ----------
if user_input:
    clean = preprocess(user_input)
    vec = tfidf.transform([clean])

    category = cat_encoder.inverse_transform(cat_model.predict(vec))[0]
    priority = pri_encoder.inverse_transform(pri_model.predict(vec))[0]

    st.subheader("Prediction")
    st.write(f"**Category:** {category}")
    st.write(f"**Priority:** {priority}")
