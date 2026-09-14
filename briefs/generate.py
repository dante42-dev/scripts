#!/usr/bin/env python3
import os
import yaml
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone
from qdrant_client import QdrantClient

BASE_DIR = "/opt/k13-ai/pipelines/daily-briefs"
CONFIG_FILE = os.path.join(BASE_DIR, "config.yaml")

client = QdrantClient(url="http://localhost:6333")

def fetch_articles(query, top_k, lookback_hours):
    results = client.search(
        collection_name="rss_articles",
        query_text=query,
        limit=top_k
    )
    return [
        {
            "title": r.payload.get("title", "Untitled"),
            "summary": r.payload.get("content", r.payload.get("summary", ""))
        }
        for r in results
    ]

def summarize(endpoint, model, prompt, articles):
    if not articles:
        return "_No new items indexed in the last 24h._\n"
    content = "\n\n".join([f"- **{a['title']}**: {a['summary']}" for a in articles])
    payload = {
        "model": model,
        "prompt": f"{prompt}\n\nArticles:\n{content}\n\nSummary:",
        "stream": False
    }
    res = requests.post(f"{endpoint}/api/generate", json=payload, timeout=120)
    res.raise_for_status()
    return res.json().get("response", "").strip()

def send_email(subject, body_markdown, email_cfg):
    if not email_cfg.get("enabled", False):
        return

    sender = email_cfg.get("smtp_user")
    recipient = email_cfg.get("recipient")
    
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"K13 Daily Briefs <{sender}>"
    msg["To"] = recipient

    # Plain text version
    msg.attach(MIMEText(body_markdown, "plain"))

    # Basic styled HTML wrapper
    html_content = f"""
    <html>
      <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #222; max-width: 680px; margin: 0 auto; padding: 20px;">
        <pre style="white-space: pre-wrap; font-family: inherit;">{body_markdown}</pre>
      </body>
    </html>
    """
    msg.attach(MIMEText(html_content, "html"))

    print(f"[*] Sending brief via SMTP to {recipient}...")
    try:
        with smtplib.SMTP(email_cfg["smtp_host"], email_cfg.get("smtp_port", 587), timeout=30) as server:
            server.starttls()
            server.login(sender, email_cfg["smtp_pass"])
            server.sendmail(sender, [recipient], msg.as_string())
        print(f"[+] Email delivered successfully.")
    except Exception as e:
        print(f"[!] SMTP send failed: {e}")

def main():
    with open(CONFIG_FILE, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg["general"].get("output_dir", os.path.join(BASE_DIR, "output"))
    os.makedirs(out_dir, exist_ok=True)
    
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_file = os.path.join(out_dir, f"brief_{date_str}.md")
    
    output = [f"# Daily Brief — {date_str}\n"]

    for sub in cfg.get("subjects", []):
        if not sub.get("enabled", True):
            continue
        print(f"[*] Compiling: {sub['title']}")
        try:
            items = fetch_articles(sub["query"], sub.get("top_k", 5), cfg["general"]["lookback_hours"])
        except Exception as e:
            print(f"[!] Qdrant query error on '{sub['title']}': {e}")
            items = []

        summary = summarize(
            cfg["general"]["ollama_endpoint"],
            sub.get("model", cfg["general"]["default_model"]),
            sub["prompt"],
            items
        )
        output.append(f"## {sub['title']}\n\n{summary}\n\n---\n")

    full_brief = "\n".join(output)

    with open(out_file, "w") as f:
        f.write(full_brief)
    print(f"[+] Output written to: {out_file}")

    if "email" in cfg:
        send_email(f"K13 Daily Brief — {date_str}", full_brief, cfg["email"])

if __name__ == "__main__":
    main()
