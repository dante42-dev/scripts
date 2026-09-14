import os
import re
import time
import requests
import psycopg2
from psycopg2.extras import execute_batch
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# --- Configuration ---
FRESHRSS_BASE_URL = os.getenv("FRESHRSS_BASE_URL", "http://hostname:9801")
FRESHRSS_USER = os.getenv("FRESHRSS_USER", "admin")
FRESHRSS_API_PASS = os.getenv("FRESHRSS_API_PASS", "password")

PG_HOST = os.getenv("PG_HOST", "hostname")
PG_PORT = int(os.getenv("PG_PORT", "5433"))
PG_DB = os.getenv("PG_DB", "fetch42")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASS = os.getenv("PG_PASS", "password")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.containers.internal:11434")
QDRANT_HOST = os.getenv("QDRANT_HOST", "host.containers.internal")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = "rss_articles"
BATCH_SIZE = 100


def init_db(pg_conn):
    schema_sql = """
    CREATE TABLE IF NOT EXISTS rss_articles (
        id VARCHAR(512) PRIMARY KEY,
        feed_title VARCHAR(255),
        title TEXT NOT NULL,
        url TEXT,
        author VARCHAR(255),
        clean_content TEXT,
        published_at TIMESTAMP WITH TIME ZONE,
        fetched_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        vector_indexed BOOLEAN DEFAULT FALSE
    );

    CREATE INDEX IF NOT EXISTS idx_rss_articles_unindexed 
        ON rss_articles(vector_indexed) WHERE vector_indexed = FALSE;
    """
    with pg_conn.cursor() as cur:
        cur.execute(schema_sql)
    pg_conn.commit()
    print(f"[Postgres] Connected to '{pg_conn.info.dbname}'. Schema verified.")


def get_freshrss_auth(session):
    login_url = f"{FRESHRSS_BASE_URL.rstrip('/')}/api/greader.php/accounts/ClientLogin"
    data = {"Email": FRESHRSS_USER, "Passwd": FRESHRSS_API_PASS}
    resp = session.post(login_url, data=data, timeout=10)
    resp.raise_for_status()

    match = re.search(r"Auth=(.+)", resp.text)
    if not match:
        raise ValueError("Could not extract Auth token from FreshRSS response.")
    return match.group(1).strip()


def get_action_token(session, auth_token):
    token_url = f"{FRESHRSS_BASE_URL.rstrip('/')}/api/greader.php/reader/api/0/token"
    headers = {"Authorization": f"GoogleLogin auth={auth_token}"}
    try:
        resp = session.get(token_url, headers=headers, timeout=5)
        if resp.status_code == 200:
            return resp.text.strip()
    except Exception as e:
        print(f"[FreshRSS] Warning: Could not fetch action token: {e}")
    return ""


def fetch_unread_articles(session, auth_token, limit=BATCH_SIZE):
    stream_url = f"{FRESHRSS_BASE_URL.rstrip('/')}/api/greader.php/reader/api/0/stream/contents/reading-list"
    headers = {"Authorization": f"GoogleLogin auth={auth_token}"}
    params = {
        "xt": "user/-/state/com.google/read",
        "n": limit
    }
    resp = session.get(stream_url, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json().get("items", [])


def mark_articles_as_read(session, auth_token, action_token, item_ids):
    if not item_ids:
        return
    edit_url = f"{FRESHRSS_BASE_URL.rstrip('/')}/api/greader.php/reader/api/0/edit-tag"
    headers = {"Authorization": f"GoogleLogin auth={auth_token}"}

    data = [
        ("a", "user/-/state/com.google/read"),
        ("async", "true"),
    ]
    if action_token:
        data.append(("T", action_token))
    for iid in item_ids:
        data.append(("i", iid))

    try:
        r = session.post(edit_url, headers=headers, data=data, timeout=15)
        if r.status_code == 200:
            print(f"[FreshRSS] Marked {len(item_ids)} articles as read.")
        else:
            print(f"[FreshRSS] Edit-tag warning ({r.status_code}): {r.text.strip()}")
    except Exception as e:
        print(f"[FreshRSS] Failed to mark articles as read: {e}")


def clean_html(raw_html):
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def get_embedding(session, text):
    url = f"{OLLAMA_HOST.rstrip('/')}/api/embeddings"
    res = session.post(url, json={"model": "nomic-embed-text", "prompt": text}, timeout=30)
    res.raise_for_status()
    return res.json()["embedding"]


def stage_to_postgres(pg_conn, items):
    records = []
    for item in items:
        item_id = item.get("id")
        title = item.get("title", "Untitled")
        author = item.get("author", "")
        feed_title = item.get("origin", {}).get("title", "")

        pub = item.get("published", time.time())
        try:
            pub_ts = float(pub)
            if pub_ts > 1e11:
                pub_ts /= 1000.0
        except (ValueError, TypeError):
            pub_ts = time.time()

        url = ""
        if item.get("canonical"):
            url = item["canonical"][0].get("href", "")
        elif item.get("alternate"):
            url = item["alternate"][0].get("href", "")

        raw_content = ""
        if "content" in item and "content" in item["content"]:
            raw_content = item["content"]["content"]
        elif "summary" in item and "content" in item["summary"]:
            raw_content = item["summary"]["content"]

        clean_content = clean_html(raw_content)

        records.append((
            item_id, feed_title, title, url, author, clean_content, pub_ts
        ))

    query = """
    INSERT INTO rss_articles (id, feed_title, title, url, author, clean_content, published_at)
    VALUES (%s, %s, %s, %s, %s, %s, to_timestamp(%s))
    ON CONFLICT (id) DO UPDATE SET
        title = EXCLUDED.title,
        clean_content = EXCLUDED.clean_content;
    """
    with pg_conn.cursor() as cur:
        execute_batch(cur, query, records)
        cur.execute("SELECT COUNT(*) FROM rss_articles;")
        total_staged = cur.fetchone()[0]
    pg_conn.commit()
    print(f"[Postgres] Staged batch of {len(records)} articles (Total rows in DB: {total_staged}).")


def drain_unvectorized_to_qdrant(session, pg_conn, q_client):
    """Processes all unindexed articles in Postgres in batches until empty."""
    if not q_client.collection_exists(QDRANT_COLLECTION):
        q_client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )

    total_vectorized = 0
    while True:
        with pg_conn.cursor() as cur:
            cur.execute("""
                SELECT id, title, feed_title, url, clean_content
                FROM rss_articles
                WHERE vector_indexed = FALSE
                LIMIT %s;
            """, (BATCH_SIZE,))
            rows = cur.fetchall()

        if not rows:
            break

        points = []
        processed_ids = []

        for item_id, title, feed_title, url, clean_content in rows:
            prompt_text = f"Title: {title}\nFeed: {feed_title}\n\n{clean_content[:1800]}"
            try:
                emb = get_embedding(session, prompt_text)
                point_id = abs(hash(item_id)) % (1 << 63)
                points.append(PointStruct(
                    id=point_id,
                    vector=emb,
                    payload={
                        "article_id": item_id,
                        "title": title,
                        "feed": feed_title,
                        "url": url,
                        "text": clean_content[:3000]
                    }
                ))
                processed_ids.append((item_id,))
            except Exception as e:
                print(f"[Error] Failed to embed article {item_id}: {e}")

        if points:
            q_client.upsert(collection_name=QDRANT_COLLECTION, points=points)
            with pg_conn.cursor() as cur:
                execute_batch(cur, "UPDATE rss_articles SET vector_indexed = TRUE WHERE id = %s;", processed_ids)
            pg_conn.commit()
            total_vectorized += len(points)
            print(f"[Qdrant] Vectorized batch of {len(points)} (Total this run: {total_vectorized}).")

    if total_vectorized == 0:
        print("[Qdrant] No pending articles to vectorize.")


def main():
    print("[Pipeline] Starting full FreshRSS backlog drain...")
    http_session = requests.Session()

    pg_conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS
    )
    init_db(pg_conn)
    q_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)

    try:
        token = get_freshrss_auth(http_session)
        action_token = get_action_token(http_session, token)

        seen_ids = set()
        batch_num = 1
        total_ingested = 0

        # Continuous loop draining FreshRSS unread queue
        while True:
            articles = fetch_unread_articles(http_session, token, limit=BATCH_SIZE)
            if not articles:
                print("[FreshRSS] Unread queue completely empty.")
                break

            current_ids = [item["id"] for item in articles if "id" in item]

            # Safeguard: detect if FreshRSS returned the same batch to prevent infinite loop
            if set(current_ids).issubset(seen_ids):
                print("[FreshRSS] Detected repeated IDs (read status pending server flush). Finishing cycle.")
                break

            seen_ids.update(current_ids)
            print(f"\n--- [Batch {batch_num}] Ingesting {len(articles)} articles ---")

            stage_to_postgres(pg_conn, articles)
            mark_articles_as_read(http_session, token, action_token, current_ids)

            total_ingested += len(articles)
            batch_num += 1
            time.sleep(0.5)

        print(f"\n[Postgres] Finished staging {total_ingested} total unread articles.")
        print("[Qdrant] Commencing vector indexing for all pending records...")
        drain_unvectorized_to_qdrant(http_session, pg_conn, q_client)

    finally:
        pg_conn.close()
    print("[Pipeline] Full backlog sync completed.")


if __name__ == "__main__":
    main()