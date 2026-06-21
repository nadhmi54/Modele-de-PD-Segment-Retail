#!/bin/bash
# Démarre l'API FastAPI et le frontend Streamlit en parallèle

echo "Démarrage API FastAPI sur port 8000..."
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 &

echo "Démarrage Frontend Streamlit sur port 8501..."
python -m streamlit run frontend/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false &

wait
