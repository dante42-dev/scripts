#!/usr/bin/env bash
set -e

COMPOSE_FILE="/opt/k13-ai/ollama/docker-compose.ollama.yaml"
CONTAINER_NAME="k13-ollama"

setup_wrapper() {
  podman exec "$CONTAINER_NAME" sh -c '
    TARGET_DIR=$(dirname $(find / -name "ollama-lib" 2>/dev/null | head -n 1))
    if [ -n "$TARGET_DIR" ] && [ ! -f /usr/local/bin/ollama ]; then
      cat << "WRAPPER" > /usr/local/bin/ollama
#!/bin/sh
cd "'"$TARGET_DIR"'"
exec "'"$TARGET_DIR"'/ollama" "$@"
WRAPPER
      chmod +x /usr/local/bin/ollama
    fi
  ' 2>/dev/null || true
}

case "$1" in
  up|start)
    echo "[Ollama] Starting container..."
    podman-compose -f "$COMPOSE_FILE" up -d
    sleep 2
    setup_wrapper
    ;;
  restart)
    echo "[Ollama] Restarting container..."
    podman-compose -f "$COMPOSE_FILE" restart
    sleep 2
    setup_wrapper
    ;;
  down|stop)
    podman-compose -f "$COMPOSE_FILE" down
    ;;
  logs)
    podman logs -f "$CONTAINER_NAME"
    ;;
  status)
    echo "=== Container Status ==="
    podman ps --filter "name=$CONTAINER_NAME"
    echo ""
    echo "=== Loaded Models ==="
    curl -s http://localhost:11434/api/ps | python3 -c '
import sys, json
data = json.load(sys.stdin).get("models", [])
if not data:
    print("No models currently loaded in memory.")
for m in data:
    size_gb = m.get("size", 0) / (1024**3)
    print(f"- {m.get(\x27name\x27)} (Size: {size_gb:.2f} GB)")
' 2>/dev/null || echo "Ollama API not responding."
    ;;
  test)
    MODEL="${2:-llama3.2}"
    echo "[Ollama] Running test inference on $MODEL..."
    setup_wrapper
    curl -s http://localhost:11434/api/generate -d "{
      \"model\": \"$MODEL\",
      \"prompt\": \"Write a 150-word summary of distributed storage systems.\",
      \"stream\": false
    }" | python3 -c '
import sys, json
res = json.load(sys.stdin)
if "error" in res:
    print("Error:", res["error"])
else:
    eval_count = res.get("eval_count", 0)
    eval_dur = res.get("eval_duration", 1) / 1e9
    tps = eval_count / eval_dur if eval_dur > 0 else 0
    preview = res.get("response", "").strip()[:140]
    print(f"\nResponse preview:\n{preview}...\n")
    print(f"Generated: {eval_count} tokens in {eval_dur:.2f}s")
    print(f"Speed    : {tps:.2f} tok/s")
'
    echo ""
    echo "=== Engine Layer Placement ==="
    podman logs "$CONTAINER_NAME" --tail 40 | grep -iE "offload|sycl0 model buffer" | tail -n 2 || true
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|logs|status|test [model]}"
    exit 1
    ;;
esac
