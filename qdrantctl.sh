#!/usr/bin/env bash
set -e

COMPOSE_FILE="/opt/k13-ai/qdrant/docker-compose.qdrant.yaml"
BACKUP_DIR="/srv/dropit/base/services/qdrant"

case "$1" in
  up|start)
    echo "[Qdrant] Starting service..."
    podman-compose -f "$COMPOSE_FILE" up -d
    ;;
  down|stop)
    echo "[Qdrant] Stopping service..."
    podman-compose -f "$COMPOSE_FILE" down
    ;;
  restart)
    echo "[Qdrant] Restarting..."
    podman-compose -f "$COMPOSE_FILE" restart
    ;;
  logs)
    podman logs -f k13-qdrant
    ;;
  status)
    podman ps --filter "name=k13-qdrant"
    echo ""
    curl -s http://localhost:6333/healthz && echo " -> Qdrant REST API OK" || echo "[Notice] Qdrant unreachable on port 6333."
    echo ""
    echo "Collections currently loaded:"
    curl -s http://localhost:6333/collections | python3 -m json.tool 2>/dev/null || true
    ;;
  snapshot)
    COLLECTION="${2:-rss_articles}"
    echo "[Qdrant] Creating snapshot for collection: $COLLECTION"
    curl -X POST "http://localhost:6333/collections/${COLLECTION}/snapshots"
    ;;
  backup)
    mkdir -p "$BACKUP_DIR"
    cp -v "$COMPOSE_FILE" "$BACKUP_DIR/"
    cp -v "$0" "$BACKUP_DIR/"
    echo "[Qdrant] Backed up configs to $BACKUP_DIR"
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|logs|status|snapshot <collection>|backup}"
    exit 1
    ;;
esac
