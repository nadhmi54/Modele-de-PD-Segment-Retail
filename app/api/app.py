# -*- coding: utf-8 -*-
"""
API REST FastAPI — Modèle PD Retail
Endpoints :
  GET  /          → documentation redirect
  GET  /health    → statut de l'API
  GET  /model-info → informations du modèle
  POST /predict   → score PD d'un client
"""

import os
import sys
import yaml
import joblib
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import predict_client
from PD_Model.app.api.schemas import ClientInput, PredictionOutput, HealthOutput, ModelInfoOutput, RiskBand

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

MODEL_DIR = os.path.join(BASE_DIR, cfg['paths']['models'])
_model    = joblib.load(os.path.join(MODEL_DIR, 'logistic_regression.pkl'))
_results  = joblib.load(os.path.join(MODEL_DIR, 'evaluation_results.pkl'))
_started  = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=cfg['api']['title'],
    version=cfg['api']['version'],
    description="""
## API de Scoring PD Retail — EY Advisory

Calcule la **Probabilité de Défaut** d'un client retail à 12 mois
à partir de ses données financières et comportementales.

### Modèle
- Algorithme : Régression Logistique (Bâle II/III)
- AUC-ROC : **0.8625** | Gini : **0.7251** | KS : **0.5775**

### Endpoints
- `POST /predict` — Score PD d'un nouveau client
- `GET  /health`  — Statut de l'API
- `GET  /model-info` — Métriques du modèle
    """,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthOutput, tags=["Monitoring"])
def health():
    """Vérifie que l'API est opérationnelle."""
    return HealthOutput(
        status="ok",
        model="Logistic Regression PD Retail",
        version=cfg['api']['version'],
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )


@app.get("/model-info", response_model=ModelInfoOutput, tags=["Monitoring"])
def model_info():
    """Retourne les métriques de performance du modèle."""
    return ModelInfoOutput(
        algorithm="LogisticRegression",
        C=float(_model.C),
        class_weight=str(_model.class_weight),
        solver=_model.solver,
        n_features=int(_model.n_features_in_),
        cv_auc_roc=0.8542,
        test_auc_roc=float(_results['auc_test']),
        gini=float(_results['gini']),
        ks_statistic=float(_results['ks_stat']),
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Scoring"])
def predict(client: ClientInput):
    """
    Calcule la Probabilité de Défaut (PD) d'un client.

    Retourne :
    - **pd_score** : probabilité de défaut entre 0 et 1
    - **pd_pct** : probabilité en pourcentage
    - **risk_band** : classe de risque (Faible / Modéré / Élevé / Très élevé)
    - **top_drivers** : variables ayant le plus contribué au score
    """
    try:
        client_data = {
            'bureau_score':        client.bureau_score,
            'monthly_income':      client.monthly_income,
            'dti':                 client.dti,
            'utilization_rate':    client.utilization_rate,
            'num_credit_lines':    client.num_credit_lines,
            'missed_payments_3m':  client.missed_payments_3m,
            'overdrawn_days_3m':   client.overdrawn_days_3m,
            'max_dpd_12m':         client.max_dpd_12m,
            'salary_domiciliation': client.salary_domiciliation,
            'employment_status':   client.employment_status.value,
        }

        result = predict_client(client_data)

        return PredictionOutput(
            pd_score    = result['pd_score'],
            pd_pct      = result['pd_pct'],
            risk_band   = RiskBand(**result['risk_band']),
            top_drivers = dict(list(result['contributions'].items())[:5]),
            timestamp   = datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host=cfg['api']['host'],
        port=cfg['api']['port'],
        reload=True
    )
