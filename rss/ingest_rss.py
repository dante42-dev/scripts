import os
import time
import hashlib
import schedule
import requests
import psycopg2
from psycopg2.extras import execute_values
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

FRESHRSS_URL = os.getenv("FRESHRSS_URL", "").rstrip("/")
FRESHRSS_API_PASSWORD = os.getenv("FRESHRSS_API_PASSWORD", "password")
FRESHRSS_USER = os.getenv("POSTGRES_USER", "freshrss") # FreshRSS username usually matches

POSTGRES_SERVER = os.getenv("POSTGRES_DB_SERVER", "hostname:5433")
POSTGRES_DB = os.getenv("POSTGRES_DB", "fetch42")
POSTGRES_USER = os.getenv("POSTGRES_USER", "freshrss")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL_MINUTES", "30"))

COLLECTION_NAME = "news_articles"

# Init Qdrant Client
q_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def get_pg_connection():
    host, port = POSTGRES_SERVER.split(":")
    return psycopg2.connect(
        host=host,
        port=port,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD
    )

def init_postgres():
    """Ensure cold storage table exists."""
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS freshrss_articles (
                    id VARCHAR(255) PRIMARY KEY,
                    title TEXT,
                    url TEXT,
                    author VARCHAR(255),
                    published_at TIMESTAMP,
                    content TEXT,
                    raw_html TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        conn.commit()
    print("[Postgres] Cold storage table verified.")

def get_freshrss_auth_token():
    """Authenticate with FreshRSS Google Reader API."""
    login_url = f"{FRESHRSS_URL}/accounts/ClientLogin"
    params = {
        "Email": FRESHRSS_USER,
        "Passwd": FRESHRSS_API_PASSWORD
    }
    res = requests.post(login_url, data=params, timeout=10)
    if res.status_code == 200:
        for line in res.text.splitlines():
            if line.startswith("Auth="):
                return line.split("Auth=")[1].strip()
    raise RuntimeError(f"Failed to authenticate with FreshRSS: {res.text}")

def get_embedding(text):
    """Generate embedding vector using Ollama."""
    try:
        res = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
            timeout=30
        )
        if res.status_code == 200:
            return res.json().get("embedding", [])
    except Exception as e:
        print(f"[Ollama] Embedding error: {e}")
    return []

def ensure_qdrant_collection(dim):
    collections = [c.name for c in q_client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        q_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
        print(f"[Qdrant] Created collection '{COLLECTION_NAME}' (dim={dim}).")

def sync_pipeline():
    print(f"\n--- [Pipeline Run: {time.strftime('%Y-%m-%d %H:%M:%S')}] ---")
    try:
        auth_token = get_freshrss_auth_token()
        headers = {"Authorization": f"GoogleLogin auth={auth_token}"}
        
        # Pull unread/recent stream
        stream_url = f"{FRESHRSS_URL}/reader/api/0/stream/contents/reading-list?output=json&n=50"
        res = requests.get(stream_url, headers=headers, timeout=15)
        
        if res.status_code != 200:
            print(f"[FreshRSS] Failed to fetch stream: {res.status_code} {res.text}")
            return
            
        items = res.json().get("items", [])
        print(f"[FreshRSS] Retrieved {len(items)} items from feed.")
        if not items:
            return

        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                for item in items:
                    article_id = item.get("id", "")
                    title = item.get("title", "No Title")
                    author = item.get("author", "")
                    published_ts = item.get("published", int(time.time()))
                    canonical_urls = item.get("canonical", [])
                    url = canonical_urls[0].get("href", "") if canonical_urls else ""
                    
                    raw_html = item.get("content", {}).get("content", "") or item.get("summary", {}).get("content", "")
                    clean_text = BeautifulSoup(raw_html, "html.parser").get_text(separator=" ", strip=True)
                    
                    # 1. Store to PostgreSQL (Cold Storage Upsert)
                    cur.execute("""
                        INSERT INTO freshrss_articles (id, title, url, author, published_at, content, raw_html)
                        VALUES (%s, %s, %s, %s, to_timestamp(%s), %s, %s)
                        ON CONFLICT (id) DO NOTHING;
                    """, (article_id, title, url, author, published_ts, clean_text, raw_html))
                    
                    # 2. Vectorize and Send to Qdrant
                    # Prepare representation blob: Title + up to 1500 chars of body
                    vector_blob = f"{title}\n\n{clean_text[:1500]}"
                    emb = get_embedding(vector_blob)
                    if emb:
                        ensure_qdrant_collection(len(emb))
                        
                        # Generate deterministic 64-bit integer ID for Qdrant from the article_id string
                        qdrant_id = int(hashlib.md5(article_id.encode('utf-8')).hexdigest()[:15], 16)
                        
                        q_client.upsert(
                            collection_name=COLLECTION_NAME,
                            points=[PointStruct(
                                id=qdrant_id,
                                vector=emb,
                                payload={
                                    "freshrss_id": article_id,
                                    "title": title,
                                    "url": url,
                                    "author": author,
                                    "published_at": published_ts,
                                    "text_snippet": clean_text[:500]
                                }
                            )]
                        )
            conn.commit()
            print(f"[Pipeline] Successfully synced {len(items)} items to PostgreSQL and Qdrant.")

    except Exception as e:
        print(f"[Pipeline Error] {e}")

if __name__ == "__main__":
    time.sleep(3) # allow DB and network stacks to fully register
    init_postgres()
    sync_pipeline()
    schedule.every(POLL_INTERVAL).minutes.do(sync_pipeline)
    
    print(f"[Scheduler] Polling every {POLL_INTERVAL} minutes.")
    while True:
        schedule.run_pending()
        time.sleep(5)