# ══════════════════════════════════════════════════════════════════════════════
# Dockerfile — Modèle PD Retail
# Image : API FastAPI + Frontend Streamlit
# ══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim

# Métadonnées
LABEL maintainer="EY Advisory"
LABEL description="PD Scoring Model — Retail Segment"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    APP_HOME=/app

WORKDIR $APP_HOME

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copie du projet
COPY . .

# Création des dossiers nécessaires
RUN mkdir -p models reports/figures mlruns

# Port API FastAPI
EXPOSE 8000
# Port Frontend Streamlit
EXPOSE 8501

# Healthcheck sur l'API
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Script de démarrage (API + Frontend)
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

CMD ["./docker-entrypoint.sh"]
