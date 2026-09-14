#!/usr/bin/env python3
import os
import re
import yaml
import requests
from qdrant_client import QdrantClient

BASE_DIR = "/opt/k13-ai/pipelines/daily-briefs"
CONFIG_FILE = os.path.join(BASE_DIR, "config.yaml")

with open(CONFIG_FILE, "r") as f:
    cfg = yaml.safe_load(f)

client = QdrantClient(url="http://localhost:6333")
collection = "rss_articles"

print(f"[*] Scrolling collection '{collection}'...")
records, _ = client.scroll(
    collection_name=collection,
    limit=1000,
    with_payload=True,
    with_vectors=False
)

ai_filter = re.compile(
    r'\b(ai|artificial intelligence|llm|llms|machine learning|deep learning|ollama|model|models|weights|gpu|neural|openai|anthropic|gemini|transformer|inference|m5|m6|quantization)\b',
    re.IGNORECASE
)

ai_articles = []
for r in records:
    payload = r.payload or {}
    title = payload.get("title", "Untitled")
    body = payload.get("content", payload.get("summary", ""))
    if ai_filter.search(f"{title} {body}"):
        ai_articles.append({"title": title, "body": body[:450]})

print(f"[+] Isolated {len(ai_articles)} AI-related articles out of {len(records)} total records.")

# Select top matches to fit comfortably within LLM context window
selected = ai_articles[:40]
context = "\n\n".join([f"- **{a['title']}**: {a['body']}" for a in selected])

prompt = (
    "You are an expert AI systems analyst. Analyze the following news articles and produce a detailed, "
    "structured intelligence brief covering the top AI topics and breakthroughs.\n\n"
    "Organize the brief into these 4 sections:\n"
    "1. **Model Releases & Architectures**: New weights, open models, benchmarks, and multimodal shifts.\n"
    "2. **Silicon & Compute Infrastructure**: Hardware developments (Nvidia, Apple M-series, TPUs), accelerators, and compute supply.\n"
    "3. **Local Tooling & Inference**: Runtime optimizations, local LLM tooling (Ollama, llama.cpp, quantization), and developer frameworks.\n"
    "4. **Governance, Security & Ecosystem**: Threats, legal decisions, safety research, and industry movements.\n\n"
    "Synthesize the information directly with concrete names and facts instead of high-level generic statements.\n\n"
    f"Articles:\n{context}\n\n"
    "AI Intelligence Brief:"
)

endpoint = cfg["general"]["ollama_endpoint"]
model = cfg["general"]["default_model"]

print(f"[*] Generating targeted AI summary with {model}...")
res = requests.post(
    f"{endpoint}/api/generate",
    json={"model": model, "prompt": prompt, "stream": False},
    timeout=600
)
res.raise_for_status()
summary = res.json().get("response", "").strip()

out_file = "/opt/k13-ai/pipelines/daily-briefs/output/test_ai_brief.md"
with open(out_file, "w") as f:
    f.write(f"# Top AI Developments Brief ({len(selected)} Articles Synthesized)\n\n{summary}\n")

print(f"[+] Saved output to {out_file}")
