# -*- coding: utf-8 -*-
"""
Pipeline de prediction PD — applique exactement le meme preprocessing que l'entrainement.
"""

import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Chargement unique des artefacts au demarrage
_model    = joblib.load(os.path.join(MODEL_DIR, 'logistic_regression.pkl'))
_scaler   = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
_medians  = joblib.load(os.path.join(MODEL_DIR, 'imputation_medians.pkl'))
_caps     = joblib.load(os.path.join(MODEL_DIR, 'capping_caps.pkl'))
_num_cols = joblib.load(os.path.join(MODEL_DIR, 'num_cols.pkl'))
_bin_cols = joblib.load(os.path.join(MODEL_DIR, 'binary_cols.pkl'))

EMPLOYMENT_CATEGORIES = ['Permanent', 'Self_Employed', 'Student', 'Temporary', 'Unemployed']

RISK_BANDS = [
    (0.00, 0.15, 'Faible',     '#27AE60', 'Approuver',         1),
    (0.15, 0.35, 'Modere',     '#F0A500', 'Approuver avec suivi', 2),
    (0.35, 0.55, 'Eleve',      '#E67E22', 'Revue manuelle',    3),
    (0.55, 1.01, 'Tres eleve', '#C0392B', 'Refuser',           4),
]


def _get_risk_band(pd_score):
    for lo, hi, label, color, reco, level in RISK_BANDS:
        if lo <= pd_score < hi:
            return {'label': label, 'color': color, 'recommendation': reco, 'level': level}
    return {'label': 'Tres eleve', 'color': '#C0392B', 'recommendation': 'Refuser', 'level': 4}


def predict_client(client_data: dict) -> dict:
    """
    client_data : dict avec les 10 variables brutes du client.
    Retourne : pd_score, risk_band, contributions, processed_features.
    """
    df = pd.DataFrame([client_data])

    # 1. Imputation mediane
    df['bureau_score']   = df['bureau_score'].fillna(_medians['bureau_score'])
    df['monthly_income'] = df['monthly_income'].fillna(_medians['monthly_income'])

    # 2. Capping P99
    for col, cap in _caps.items():
        df[col] = df[col].clip(upper=cap)

    # 3. One-Hot Encoding employment_status
    emp = df['employment_status'].iloc[0]
    df = df.drop(columns=['employment_status'])
    for cat in EMPLOYMENT_CATEGORIES[1:]:  # ref = Permanent
        df[f'employment_status_{cat}'] = 1 if emp == cat else 0

    # 4. Reordonner colonnes exactement comme lors de l'entrainement
    feature_names = _model.feature_names_in_.tolist()
    for c in feature_names:
        if c not in df.columns:
            df[c] = 0
    df = df[feature_names]

    # 5. Standardisation (num_cols uniquement)
    df_sc = df.copy()
    df_sc[_num_cols] = _scaler.transform(df[_num_cols])

    # 6. Prediction
    pd_score = float(_model.predict_proba(df_sc)[0, 1])
    risk     = _get_risk_band(pd_score)

    # 7. Contributions (coef x valeur standardisee)
    coefs    = _model.coef_[0]
    contribs = {col: float(coef * df_sc[col].iloc[0])
                for col, coef in zip(feature_names, coefs)}
    contribs_sorted = dict(sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True))

    return {
        'pd_score':      round(pd_score, 4),
        'pd_pct':        round(pd_score * 100, 2),
        'risk_band':     risk,
        'recommendation': risk['recommendation'],
        'top_drivers':   contribs_sorted,
        'contributions': contribs_sorted,
        'features_raw':  client_data,
    }
