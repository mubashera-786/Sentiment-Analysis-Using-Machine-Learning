🔧 Prerequisites
Make sure you have the necessary libraries installed:

pip install textblob nltk
python -m textblob.download_corpora

🧠 Simple Sentiment Analysis Code (Using TextBlob)

from textblob import TextBlob

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity

    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

# Example
if __name__ == "__main__":
    user_input = input("Enter a sentence: ")
    result = analyze_sentiment(user_input)
    print(f"Sentiment: {result}")
