💡 Project: Sentiment Analysis on Movie Reviews

🔧 Tools & Libraries
Python 3.x
pandas, sklearn, nltk, TextBlob

(optional) Flask or Streamlit for web app

📁 Project Structure

sentiment-analysis-project/
│
├── dataset/
│   └── movie_reviews.csv
├── sentiment_model.py
├── predict.py
├── app.py                # Optional: Flask or Streamlit app
├── requirements.txt
└── README.md

📂 Step 1: Dataset
Use this free dataset:

IMDb Movie Reviews: Kaggle Dataset

Download and save as movie_reviews.csv.

🧹 Step 2: Preprocessing & Training (sentiment_model.py)
python

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load data
df = pd.read_csv('dataset/movie_reviews.csv')

# Basic cleaning
df = df[['review', 'sentiment']]
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Vectorize text
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump((vectorizer, model), f)
🧪 Step 3: Prediction Script (predict.py)
python

import pickle

def predict_sentiment(text):
    with open('model.pkl', 'rb') as f:
        vectorizer, model = pickle.load(f)

    vec_text = vectorizer.transform([text])
    prediction = model.predict(vec_text)[0]
    return "Positive" if prediction == 1 else "Negative"

# Test
if __name__ == "__main__":
    user_input = input("Enter your review: ")
    result = predict_sentiment(user_input)
    print("Predicted Sentiment:", result)
🌐 Step 4 (Optional): Deploy with Streamlit (app.py)
python
Copy code
import streamlit as st
from predict import predict_sentiment

st.title("🎬 Movie Review Sentiment Analyzer")

user_input = st.text_area("Enter a movie review:")
if st.button("Analyze"):
    result = predict_sentiment(user_input)
    st.write(f"🧠 Sentiment: **{result}**")
Run with:

streamlit run app.py
📦 requirements.txt

pandas
scikit-learn
nltk
streamlit


✅ Output
Accuracy report in terminal

predict.py lets you test any sentence

app.py gives a simple web interface

