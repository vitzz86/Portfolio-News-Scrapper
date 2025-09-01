from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time
import random

# ==============================
# Initialize FastAPI App
# ==============================
app = FastAPI(
    title="Advanced News Scraper & Analysis API",
    description="An API to scrape Google News, then perform sentiment and topic analysis using transformer models.",
    version="3.2.0" # Version updated for robust scraping
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ==============================
# Load Models (once on startup)
# ==============================
print("Loading sentiment analysis models... This may take a moment.")
finbert_name = "ProsusAI/finbert"
indobert_name = "w11wo/indonesian-roberta-base-sentiment-classifier"

finbert_tokenizer = AutoTokenizer.from_pretrained(finbert_name)
finbert_model = AutoModelForSequenceClassification.from_pretrained(finbert_name)

indobert_tokenizer = AutoTokenizer.from_pretrained(indobert_name)
indobert_model = AutoModelForSequenceClassification.from_pretrained(indobert_name)
print("Models loaded successfully.")


# ==============================
# Web Scraper Module - UPGRADED
# ==============================

# --- User-Agent Rotation ---
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/117.0',
]

# --- Proxy Configuration ---
# To use proxies, get a list of them and format them like: ['http://user:pass@ip:port']
# Using proxies is the most effective way to prevent 429 errors.

# --- BEFORE (empty list) ---
# PROXY_LIST = [] 

# --- AFTER (populated with your proxies) ---
PROXY_LIST = [
]

def get_random_proxy():
    if not PROXY_LIST:
        return None
    proxy = random.choice(PROXY_LIST)
    return {'http': proxy, 'https': proxy}


MONTH_MAP = {"Jan": "Jan", "Feb": "Feb", "Mar": "Mar", "Apr": "Apr", "Mei": "May", "Jun": "Jun", "Jul": "Jul", "Agu": "Aug", "Sep": "Sep", "Okt": "Oct", "Nov": "Nov", "Des": "Dec"}

def normalize_month(text):
    for indo, eng in MONTH_MAP.items():
        text = text.replace(indo, eng)
    return text

def parse_news_date(date_text, indo=False):
    now = datetime.now()
    if not date_text: return "Unknown"
    date_text = date_text.strip()

    relative_match = re.match(r"(\d+)\s+(hour|day|week|month|year)s?\s+ago", date_text, re.I)
    if relative_match:
        value, unit = int(relative_match.group(1)), relative_match.group(2).lower()
        if "hour" in unit: return (now - timedelta(hours=value)).strftime("%Y-%m-%d")
        if "day" in unit: return (now - timedelta(days=value)).strftime("%Y-%m-%d")
        if "week" in unit: return (now - timedelta(weeks=value)).strftime("%Y-%m-%d")
        if "month" in unit: return (now - timedelta(days=30 * value)).strftime("%Y-%m-%d")
        if "year" in unit: return (now - timedelta(days=365 * value)).strftime("%Y-%m-%d")

    if indo:
        indo_match = re.match(r"(\d+)\s+(jam|hari|minggu|bulan|tahun)\s+lalu", date_text, re.I)
        if indo_match:
            value, unit = int(indo_match.group(1)), indo_match.group(2).lower()
            if unit == "jam": return (now - timedelta(hours=value)).strftime("%Y-%m-%d")
            if unit == "hari": return (now - timedelta(days=value)).strftime("%Y-%m-%d")
            if unit == "minggu": return (now - timedelta(weeks=value)).strftime("%Y-%m-%d")
            if unit == "bulan": return (now - timedelta(days=30 * value)).strftime("%Y-%m-%d")
            if unit == "tahun": return (now - timedelta(days=365 * value)).strftime("%Y-%m-%d")
    
    fixed_text = normalize_month(date_text)
    for fmt in ("%b %d, %Y", "%d %b %Y", "%b %d %Y", "%d %b, %Y", "%Y-%m-%d", "%d %B %Y"):
        try:
            return datetime.strptime(fixed_text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return "Unknown"
    
def scrape_google_news(keyword, num_results=100, indo=False, start_date=None, end_date=None):
    base_filter = "(business OR startup OR investment OR funding OR bisnis OR pendanaan)"
    query = f'"{keyword}" AND {base_filter}'
    url = f"https://www.google.com/search?q={query}&tbm=nws&num={num_results}"
    if indo: url += "&gl=id&hl=id"

    if start_date and end_date:
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").strftime("%m/%d/%Y")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").strftime("%m/%d/%Y")
            url += f"&tbs=cdr:1,cd_min:{start_dt},cd_max:{end_dt}"
        except Exception: pass

    response = None
    for attempt in range(4): # Increased retries
        try:
            headers = {'User-Agent': random.choice(USER_AGENTS)}
            proxies = get_random_proxy()
            
            print(f"Attempt {attempt+1} for '{keyword}' using UA: ...{headers['User-Agent'][-30:]}")
            if proxies:
                print(f"Using proxy: {proxies['http']}")

            response = requests.get(url, headers=headers, proxies=proxies, timeout=20)
            
            if response.status_code == 429:
                wait_time = random.uniform(5, 10) * (attempt + 1)
                print(f"⚠️ Rate limit hit. Waiting {wait_time:.2f}s...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            
            if "Our systems have detected unusual traffic" in response.text:
                print("❌ Captcha page detected. Skipping this request.")
                time.sleep(random.uniform(10, 20))
                return []

            break
            
        except requests.RequestException as e:
            print(f"Scraping failed for '{keyword}': {e}")
            if attempt < 3:
                time.sleep(random.uniform(3, 7))
            else:
                return []
    
    if response is None:
        print(f"❌ Failed to scrape for '{keyword}' after multiple retries.")
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    news_results = []
    for el in soup.select("div.SoaBEf"):
        try:
            title = el.select_one("div.MBeuO").get_text(strip=True)
            summary = el.select_one(".GI74Re").get_text(strip=True)
            date_text = el.select_one(".LfVVr").get_text(strip=True)
            source = el.select_one(".NUnG9d span").get_text(strip=True)
            link = el.find("a")["href"]

            if keyword.lower() not in f"{title} {summary}".lower():
                continue

            news_results.append({
                "title": title, "summary": summary, "date": parse_news_date(date_text, indo=indo),
                "source": source, "url": link
            })
        except Exception:
            continue
    return news_results

# ==============================
# Sentiment Analysis Module
# ==============================
from config import negative_keywords, positive_keywords, topic_keywords

LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
indo_stopwords = {"yang","dan","atau","tidak","ini","itu","saya","kita","kami","dengan","untuk","akan"}

def detect_language(text: str) -> str:
    words = set(re.findall(r"\w+", text.lower()))
    return "ID" if words & indo_stopwords else "EN"

def predict_sentiment(model, tokenizer, text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
    idx = scores.argmax()
    return LABELS[idx], float(scores[idx])

def get_topic(text: str) -> str:
    text_lower = text.lower()
    for topic, kws in topic_keywords.items():
        if any(kw in text_lower for kw in kws):
            return topic
    return "GENERAL"
    
def get_news_sentiment(text: str):
    if not isinstance(text, str) or not text.strip():
        return "NEUTRAL", 0.0, "GENERAL"

    lang = detect_language(text)
    sentiment, score = predict_sentiment(indobert_model, indobert_tokenizer, text) if lang == "ID" else predict_sentiment(finbert_model, finbert_tokenizer, text)
    
    if score < 0.6: 
        sentiment = "NEUTRAL"

    text_lower = text.lower()
    if any(kw in text_lower for kw in negative_keywords):
        sentiment = "NEGATIVE"; score = max(score, 0.75)
    elif any(kw in text_lower for kw in positive_keywords):
        sentiment = "POSITIVE"; score = max(score, 0.75)

    topic = get_topic(text)
    return sentiment, score, topic

# ==============================
# API Endpoint
# ==============================
@app.get("/search")
async def search_news(query: str, start_date: str = None, end_date: str = None):
    search_queries = [q.strip().lower() for q in query.split(',') if q.strip()]
    all_raw_news = []
    
    for sq in search_queries:
        print(f"Scraping Indonesian news for '{sq}'...")
        all_raw_news.extend(scrape_google_news(sq, indo=True, start_date=start_date, end_date=end_date))
        time.sleep(random.uniform(1, 2))
        
        print(f"Scraping English news for '{sq}'...")
        all_raw_news.extend(scrape_google_news(sq, indo=False, start_date=start_date, end_date=end_date))
        time.sleep(random.uniform(1.5, 3.5))

    seen, unique_news = set(), []
    for article in all_raw_news:
        key = (article['title'], article['url'])
        if key not in seen:
            seen.add(key)
            unique_news.append(article)
            
    processed_articles = []
    for article in unique_news:
        text_to_analyze = f"{article['title']}. {article['summary']}"
        sentiment, score, topic = get_news_sentiment(text_to_analyze)
        
        article_data = article.copy()
        article_data.update({
            "sentiment": sentiment,
            "sentiment_score": round(score, 4),
            "topic": topic
        })
        processed_articles.append(article_data)

    processed_articles.sort(key=lambda x: x['date'] if x['date'] != "Unknown" else "0000-00-00", reverse=True)
    
    print(f"Found and processed {len(processed_articles)} unique articles for query: '{query}'")
    return processed_articles