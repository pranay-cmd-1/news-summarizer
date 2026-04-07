from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import re
import nltk
import numpy as np
import os
import logging
import networkx as nx

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from textblob import TextBlob

from datetime import datetime
from dotenv import load_dotenv

# =========================
# ⚙️ SETUP
# =========================
load_dotenv()

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("NEWS_API_KEY")

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

domains = {
    "t20": "T20 Cricket World Cup",
    "trade": "US China Trade War",
    "quantum": "Quantum Computing research",
    "space": "NASA OR ISRO OR SpaceX",
    "nobel": "Nobel Prize winners OR Nobel Prize news"
}


@app.route("/")
def home():
    return "Backend Running 🚀"


# =========================
# 🧹 CLEANING
# =========================
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


# =========================
# 🧠 TEXTRANK
# =========================
def textrank_summary(text):
    sentences = sent_tokenize(text)

    if len(sentences) <= 2:
        return text

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences).toarray()

    sim_matrix = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                norm = np.linalg.norm(X[i]) * np.linalg.norm(X[j])
                if norm > 0:
                    sim_matrix[i][j] = np.dot(X[i], X[j]) / norm

    graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(graph)

    top = sorted(scores, key=scores.get, reverse=True)[:3]
    top = sorted(top)

    return " ".join([sentences[i] for i in top])


# =========================
# 🔑 KEYWORDS
# =========================
def extract_keywords(text, n=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])

    scores = zip(vectorizer.get_feature_names_out(), X.toarray()[0])
    sorted_words = sorted(scores, key=lambda x: x[1], reverse=True)

    return [w for w, _ in sorted_words[:n]]


# =========================
# 😊 SENTIMENT
# =========================
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    return "Neutral"


# =========================
# 🏷️ CLUSTER NAMING
# =========================
def generate_cluster_names(texts, labels):
    cluster_names = {}

    for cluster_id in set(labels):
        if cluster_id == -1:
            cluster_names[cluster_id] = "Other"
            continue

        cluster_texts = [
            texts[i] for i in range(len(texts)) if labels[i] == cluster_id
        ]

        combined_text = " ".join(cluster_texts)

        vectorizer = TfidfVectorizer(stop_words='english', max_features=3)
        X = vectorizer.fit_transform([combined_text])

        keywords = vectorizer.get_feature_names_out()

        name = " / ".join([word.capitalize() for word in keywords])

        cluster_names[cluster_id] = name

    return cluster_names


# =========================
# 📰 MAIN API
# =========================
@app.route("/get-news")
def get_news():
    domain = request.args.get("domain")

    if domain not in domains:
        return jsonify([])

    if not API_KEY:
        return jsonify([])

    url = f"https://newsapi.org/v2/everything?q={domains[domain]}&pageSize=20&apiKey={API_KEY}"

    try:
        response = requests.get(url).json()
    except:
        return jsonify([])

    if response.get("status") != "ok":
        return jsonify([])

    articles = []
    cleaned_texts = []

    for item in response.get("articles", [])[:20]:
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

    if not articles:
        return jsonify([])

    # =========================
    # 🔥 DYNAMIC CLUSTERING
    # =========================
    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_df=0.7,
            ngram_range=(1, 3)
        )

        X = vectorizer.fit_transform(cleaned_texts)
        X = normalize(X)

        if len(cleaned_texts) < 3:
            labels = [0] * len(cleaned_texts)
            total_clusters = 1
        else:
            best_k = 2
            best_score = -1

            for k in range(2, min(6, len(cleaned_texts))):
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    temp_labels = kmeans.fit_predict(X)

                    if len(set(temp_labels)) > 1:
                        score = silhouette_score(X, temp_labels)

                        if score > best_score:
                            best_score = score
                            best_k = k
                except:
                    continue

            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            total_clusters = best_k

            logger.info(f"Best K: {best_k}, Score: {best_score}")

    except Exception as e:
        logger.error(e)
        labels = [0] * len(articles)
        total_clusters = 1

    # ✅ Generate cluster names
    cluster_names = generate_cluster_names(cleaned_texts, labels)

    # =========================
    # 📦 OUTPUT
    # =========================
    final_output = []

    for i, article in enumerate(articles):
        try:
            date = datetime.strptime(
                article["publishedAt"], "%Y-%m-%dT%H:%M:%SZ"
            ).strftime("%d %b %Y")
        except:
            date = "Unknown"

        final_output.append({
            "summary": textrank_summary(article["original"]),
            "keywords": extract_keywords(article["original"]),
            "sentiment": get_sentiment(article["original"]),
            "cluster": int(labels[i]),
            "cluster_name": cluster_names.get(int(labels[i]), "General"),
            "url": article["url"],
            "date": date,
            "source": article["source"],
            "total_clusters": total_clusters
        })

    return jsonify(final_output)


# =========================
# 🚀 RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)