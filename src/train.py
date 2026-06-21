# -*- coding: utf-8 -*-
"""
Entraînement du modèle de Probabilité de Défaut — Régression Logistique
"""

import os
import sys
import yaml
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR   = os.path.join(BASE_DIR, 'models')
FIG_DIR     = os.path.join(BASE_DIR, 'reports', 'figures')
CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

mlflow_db = os.path.join(BASE_DIR, "mlruns", "mlflow.db")
os.makedirs(os.path.join(BASE_DIR, "mlruns"), exist_ok=True)
mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
mlflow.set_experiment(cfg['mlflow']['experiment_name'])

# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES PRÉPARÉES
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("CHARGEMENT DES DONNÉES PRÉPARÉES")
print("=" * 70)

X_train, X_test, y_train, y_test = joblib.load(os.path.join(MODEL_DIR, 'train_test_data.pkl'))

print(f"Train : {X_train.shape[0]:,} observations x {X_train.shape[1]} features")
print(f"Test  : {X_test.shape[0]:,} observations x {X_test.shape[1]} features")
print(f"Taux de défaut train : {y_train.mean()*100:.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# ENTRAÎNEMENT DU MODÈLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ENTRAÎNEMENT — RÉGRESSION LOGISTIQUE")
print("=" * 70)

model = LogisticRegression(
    C=1.0,
    class_weight='balanced',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

print("Paramètres du modèle :")
print(f"  C              : {model.C}")
print(f"  class_weight   : {model.class_weight}")
print(f"  solver         : {model.solver}")
print(f"  Convergence    : {model.n_iter_[0]} itérations")
print(f"\n[OK] Modèle entraîné avec succès.")

# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION CROISÉE (5-Fold Stratifiée)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("VALIDATION CROISÉE — 5-FOLD STRATIFIÉE")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
cv_f1  = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')

print(f"AUC-ROC par fold : {[f'{v:.4f}' for v in cv_auc]}")
print(f"  Moyenne : {cv_auc.mean():.4f}  |  Std : {cv_auc.std():.4f}")
print(f"\nF1-Score par fold : {[f'{v:.4f}' for v in cv_f1]}")
print(f"  Moyenne : {cv_f1.mean():.4f}  |  Std : {cv_f1.std():.4f}")

print("""
[INTERPRÉTATION]
  La validation croisée 5-fold divise le train en 5 parties :
  à chaque tour, 4 parties servent à entraîner et 1 à valider.
  On répète 5 fois et on moyenne les scores.

  Cela permet de vérifier que le modèle est STABLE :
  si les scores sont proches entre les folds (faible std),
  le modèle generalise bien et n'est pas sensible au découpage des données.
""")

# ══════════════════════════════════════════════════════════════════════════════
# COEFFICIENTS DU MODÈLE
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("COEFFICIENTS DU MODÈLE")
print("=" * 70)

feature_names = X_train.columns.tolist()
coefficients  = model.coef_[0]
odds_ratios   = np.exp(coefficients)

coef_df = pd.DataFrame({
    'Variable'   : feature_names,
    'Coefficient': coefficients.round(4),
    'Odds Ratio' : odds_ratios.round(4)
}).sort_values('Coefficient', ascending=False)

print(coef_df.to_string(index=False))

# Graphique des coefficients
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in coef_df['Coefficient']]
bars = ax.barh(coef_df['Variable'], coef_df['Coefficient'], color=colors, edgecolor='white', alpha=0.85)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Coefficient (après standardisation)', fontsize=11)
ax.set_title('Coefficients de la Régression Logistique\n(rouge = facteur de risque | vert = facteur protecteur)',
             fontweight='bold', fontsize=12)
for bar, v in zip(bars, coef_df['Coefficient']):
    ax.text(v + (0.02 if v >= 0 else -0.02),
            bar.get_y() + bar.get_height()/2,
            f'{v:.3f}', va='center',
            ha='left' if v >= 0 else 'right', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '13_coefficients.png'), bbox_inches='tight')
plt.close()
print("\nGraphique 13_coefficients.png sauvegardé.")

print("""
[INTERPRÉTATION]
  Chaque coefficient représente l'impact d'une variable sur le log-odds de défaut.
  Grâce à la standardisation, les coefficients sont COMPARABLES entre eux.

  Variables de RISQUE (coefficient positif → rouge) :
    Un coefficient positif signifie que plus la variable augmente,
    plus la probabilité de défaut augmente.

  Variables PROTECTRICES (coefficient négatif → vert) :
    Un coefficient négatif signifie que plus la variable augmente,
    plus la probabilité de défaut diminue.

  L'Odds Ratio = exp(coefficient) :
    OR > 1 → facteur de risque
    OR < 1 → facteur protecteur
    OR = 1.5 → cette variable multiplie les odds de défaut par 1.5
""")

# ══════════════════════════════════════════════════════════════════════════════
# MLFLOW — TRACKING DE L'EXPÉRIENCE
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("MLFLOW — TRACKING")
print("=" * 70)

with mlflow.start_run(run_name="LogisticRegression_v1"):
    # Paramètres
    mlflow.log_param("C",            model.C)
    mlflow.log_param("class_weight", model.class_weight)
    mlflow.log_param("solver",       model.solver)
    mlflow.log_param("max_iter",     model.max_iter)
    mlflow.log_param("n_features",   X_train.shape[1])
    mlflow.log_param("train_size",   X_train.shape[0])

    # Métriques CV
    mlflow.log_metric("cv_auc_mean", round(cv_auc.mean(), 4))
    mlflow.log_metric("cv_auc_std",  round(cv_auc.std(),  4))
    mlflow.log_metric("cv_f1_mean",  round(cv_f1.mean(),  4))
    mlflow.log_metric("cv_f1_std",   round(cv_f1.std(),   4))

    # Modèle
    mlflow.sklearn.log_model(model, artifact_path="model")

    run_id = mlflow.active_run().info.run_id
    print(f"Run ID   : {run_id}")
    print(f"AUC-ROC  : {cv_auc.mean():.4f}")
    print(f"F1-Score : {cv_f1.mean():.4f}")
    print("Expérience trackée dans mlruns/")

# ══════════════════════════════════════════════════════════════════════════════
# SAUVEGARDE DU MODÈLE
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SAUVEGARDE DU MODÈLE")
print("=" * 70)

joblib.dump(model, os.path.join(MODEL_DIR, 'logistic_regression.pkl'))
print("Modèle sauvegardé → models/logistic_regression.pkl")

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║              SYNTHÈSE ENTRAÎNEMENT — MODÈLE PRÊT                     ║
╠══════════════════════════════════════════════════════════════════════╣
║  Algorithme    : Régression Logistique                               ║
║  C             : 1.0   |  class_weight : balanced                    ║
║  CV AUC-ROC    : {cv_auc.mean():.4f}  ±  {cv_auc.std():.4f}                        ║
║  CV F1-Score   : {cv_f1.mean():.4f}  ±  {cv_f1.std():.4f}                        ║
║  Convergence   : {model.n_iter_[0]} itérations                              ║
║                                                                      ║
║  PROCHAINE ÉTAPE : evaluate.py — Évaluation sur le jeu de test       ║
╚══════════════════════════════════════════════════════════════════════╝
""")
