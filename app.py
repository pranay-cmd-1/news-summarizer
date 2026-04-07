from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import re
import nltk
import numpy as np
import os

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from datetime import datetime

app = Flask(__name__)

# ✅ Allow all origins (fix CORS issue)
CORS(app)

# ✅ Load API key
API_KEY = os.getenv("NEWS_API_KEY")

# ✅ NLTK setup (safe for Render)
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# ✅ ROOT ROUTE (fix 404 issue)
@app.route("/")
def home():
    return "News Summarizer Backend Running 🚀"


domains = {
    "t20": "T20 Cricket World Cup",
    "trade": "US China Trade War",
    "quantum": "Quantum Computing research",
    "space": "NASA OR ISRO OR SpaceX",
    "nobel": "Nobel Prize winners OR Nobel Prize news"
}


def clean_news_text(text):
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'REUTERS|Reuters|NEW DELHI|AP NEWS', '', text)
    return text.strip()


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


def extractive_summary(text):
    if not text.strip():
        return "No meaningful content available."

    try:
        sentences = sent_tokenize(text)
    except:
        return text

    if len(sentences) <= 2:
        return text

    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(sentences)

        scores = np.array(X.mean(axis=1)).flatten()

        top_indices = scores.argsort()[-3:][::-1]
        top_indices = sorted(top_indices)

        return " ".join([sentences[i] for i in top_indices])

    except:
        return text


def extract_keywords(text, n=5):
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform([text])

        scores = zip(vectorizer.get_feature_names_out(), X.toarray()[0])
        sorted_words = sorted(scores, key=lambda x: x[1], reverse=True)

        return [word for word, score in sorted_words[:n]]

    except:
        return []


@app.route("/get-news")
def get_news():

    domain = request.args.get("domain")

    if domain not in domains:
        return jsonify({"error": "Invalid domain"})

    url = f"https://newsapi.org/v2/everything?q={domains[domain]}&apiKey={API_KEY}"
    response = requests.get(url).json()

    if response.get("status") != "ok":
        return jsonify({"error": "News API failed", "details": response})

    articles = []
    cleaned_texts = []

    for item in response.get("articles", [])[:10]:

        title = item.get("title") or ""
        desc = item.get("description") or ""

        if not title and not desc:
            continue

        text = clean_news_text(title + ". " + desc)

        articles.append({
            "original": text,
            "url": item.get("url"),
            "publishedAt": item.get("publishedAt"),
            "source": item.get("source", {}).get("name", "Unknown")
        })

        cleaned_texts.append(preprocess(text))

    if len(articles) == 0:
        return jsonify([{
            "summary": "No news available.",
            "keywords": [],
            "cluster": 0,
            "total_clusters": 1
        }])

    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, ngram_range=(1, 2))
        X = vectorizer.fit_transform(cleaned_texts)

        if len(cleaned_texts) < 3:
            labels = [0] * len(cleaned_texts)
            best_k = 1
        else:
            best_k = 2
            best_score = -1

            for k in range(2, min(6, len(cleaned_texts))):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                temp_labels = kmeans.fit_predict(X)

                score = silhouette_score(X, temp_labels)

                if score > best_score:
                    best_score = score
                    best_k = k

            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

    except:
        labels = [0] * len(articles)
        best_k = 1

    final_output = []

    for i, article in enumerate(articles):
        try:
            date = datetime.strptime(
                article["publishedAt"],
                "%Y-%m-%dT%H:%M:%SZ"
            ).strftime("%d %b %Y")
        except:
            date = "Unknown"

        final_output.append({
            "summary": extractive_summary(article["original"]),
            "keywords": extract_keywords(article["original"]),
            "cluster": int(labels[i]),
            "url": article["url"],
            "date": date,
            "source": article["source"],
            "total_clusters": best_k
        })

    return jsonify(final_output)


if __name__ == "__main__":
    app.run()
