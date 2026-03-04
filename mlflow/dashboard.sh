#!/bin/bash
# Lance le dashboard MLflow.
# Usage: ./mlflow_dashboard.sh [port]
# Par défaut: http://127.0.0.1:5000

PORT="${1:-5000}"
echo "Dashboard: http://127.0.0.1:${PORT}  (Ctrl+C pour arrêter)"
mlflow ui --host 127.0.0.1 --port "$PORT"
