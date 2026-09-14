#!/usr/bin/env python3
import os
import yaml
import requests
from qdrant_client import QdrantClient

BASE_DIR = "/opt/k13-ai/pipelines/daily-briefs"
CONFIG_FILE = os.path.join(BASE_DIR, "config.yaml")

with open(CONFIG_FILE, "r") as f:
    cfg = yaml.safe_load(f)

client = QdrantClient(url="http://localhost:6333")
collection = "rss_articles"

print(f"[*] Scrolling entire contents of collection '{collection}'...")

records, _ = client.scroll(
    collection_name=collection,
    limit=1000,
    with_payload=True,
    with_vectors=False
)

print(f"[+] Retrieved {len(records)} total records from Qdrant.")

if not records:
    print("[!] No records found in the collection.")
    exit(0)

articles = []
for r in records:
    payload = r.payload or {}
    title = payload.get("title", "Untitled")
    body = payload.get("content", payload.get("summary", ""))
    articles.append(f"- **{title}**: {body[:400]}")

context = "\n\n".join(articles)
prompt = (
    f"You are an executive intelligence analyst. Review the following {len(records)} articles "
    "from the database and produce a comprehensive executive brief. Group the findings by main "
    "themes, highlight key takeaways, and note any significant local or technical trends:\n\n"
    f"{context}\n\nFull Database Brief:"
)

endpoint = cfg["general"]["ollama_endpoint"]
model = cfg["general"]["default_model"]

print(f"[*] Sending {len(records)} articles to Ollama ({model})...")
res = requests.post(
    f"{endpoint}/api/generate",
    json={"model": model, "prompt": prompt, "stream": False},
    timeout=600
)
res.raise_for_status()
summary = res.json().get("response", "").strip()

out_dir = cfg["general"].get("output_dir", os.path.join(BASE_DIR, "output"))
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "test_full_brief.md")

with open(out_file, "w") as f:
    f.write(f"# Complete Database Brief ({len(records)} Articles)\n\n{summary}\n")

print(f"[+] Full brief written to: {out_file}")
