# -*- coding: utf-8 -*-
"""
Préparation des données — Modèle PD Retail
Étapes : copie sécurité → suppression ID → imputation → capping
         → encodage OHE → split train/test → standardisation
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── Chemins ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'simulated_retail_pd_dataset_10_variables.xlsx')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("CHARGEMENT DES DONNÉES")
print("=" * 70)

df_raw  = pd.read_excel(DATA_PATH)
df      = df_raw.copy()

print(f"Dimensions : {df_raw.shape[0]:,} lignes x {df_raw.shape[1]} colonnes")
print("[OK] Copie de sécurité df_raw conservée.")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ÉTAPE 1 — SUPPRESSION DE L'IDENTIFIANT CLIENT")
print("=" * 70)

df.drop(columns=['customer_id'], inplace=True)
print(f"customer_id supprimé. Dimensions : {df.shape[0]:,} x {df.shape[1]}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ÉTAPE 2 — IMPUTATION DES VALEURS MANQUANTES")
print("=" * 70)

median_bureau = df['bureau_score'].median()
median_income = df['monthly_income'].median()

df['bureau_score'].fillna(median_bureau, inplace=True)
df['monthly_income'].fillna(median_income, inplace=True)

print(f"bureau_score  imputé par médiane : {median_bureau:.1f}")
print(f"monthly_income imputé par médiane : {median_income:.1f}")
print(f"Valeurs manquantes restantes : {df.isnull().sum().sum()}")

joblib.dump({'bureau_score': median_bureau, 'monthly_income': median_income},
            os.path.join(MODEL_DIR, 'imputation_medians.pkl'))

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ÉTAPE 3 — CAPPING AU 99e CENTILE")
print("=" * 70)

cols_to_cap = ['monthly_income', 'dti', 'utilization_rate']
caps = {}

print(f"{'Variable':<30} {'Max avant':>12} {'Cap P99':>12} {'Max après':>12}")
print("-" * 70)
for col in cols_to_cap:
    cap_val    = df[col].quantile(0.99)
    max_before = df[col].max()
    df[col]    = df[col].clip(upper=cap_val)
    caps[col]  = cap_val
    print(f"  {col:<28} {max_before:>12.2f} {cap_val:>12.2f} {df[col].max():>12.2f}")

joblib.dump(caps, os.path.join(MODEL_DIR, 'capping_caps.pkl'))

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ÉTAPE 4 — ENCODAGE ONE-HOT DE employment_status")
print("=" * 70)

df = pd.get_dummies(df, columns=['employment_status'], drop_first=True)
ohe_cols = [c for c in df.columns if 'employment_status' in c]
print(f"Variables créées : {ohe_cols}")
print(f"Catégorie de référence : Permanent")
print(f"Dimensions après OHE : {df.shape[0]:,} x {df.shape[1]}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ÉTAPE 5 — SPLIT TRAIN / TEST (80% / 20%)")
print("=" * 70)

TARGET = 'default_12m'
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train : {X_train.shape[0]:,} obs. | Taux défaut : {y_train.mean()*100:.2f}%")
print(f"Test  : {X_test.shape[0]:,} obs. | Taux défaut : {y_test.mean()*100:.2f}%")
print(f"Features : {X_train.shape[1]}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ÉTAPE 6 — STANDARDISATION (StandardScaler)")
print("=" * 70)

binary_cols = [c for c in X_train.columns if 'employment_status' in c] + ['salary_domiciliation']
num_cols    = [c for c in X_train.columns if c not in binary_cols]

scaler = StandardScaler()
X_train_sc = X_train.copy()
X_test_sc  = X_test.copy()

X_train_sc[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_sc[num_cols]  = scaler.transform(X_test[num_cols])

print(f"Variables standardisées ({len(num_cols)}) : {num_cols}")
print(f"Variables binaires non standardisées ({len(binary_cols)}) : {binary_cols}")
print(f"Moyenne moyenne après scaling : {X_train_sc[num_cols].mean().mean():.6f} (attendu = 0)")
print(f"Std moyenne après scaling     : {X_train_sc[num_cols].std().mean():.6f} (attendu = 1)")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SAUVEGARDE DES ARTEFACTS")
print("=" * 70)

joblib.dump((X_train_sc, X_test_sc, y_train, y_test),
            os.path.join(MODEL_DIR, 'train_test_data.pkl'))
joblib.dump(scaler,   os.path.join(MODEL_DIR, 'scaler.pkl'))
joblib.dump(df_raw,   os.path.join(MODEL_DIR, 'df_raw_backup.pkl'))
joblib.dump(num_cols, os.path.join(MODEL_DIR, 'num_cols.pkl'))
joblib.dump(binary_cols, os.path.join(MODEL_DIR, 'binary_cols.pkl'))

print("train_test_data.pkl  → X_train, X_test, y_train, y_test")
print("scaler.pkl           → StandardScaler fitté sur le train")
print("imputation_medians.pkl, capping_caps.pkl, df_raw_backup.pkl")

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║              SYNTHÈSE PREPROCESSING — DONNÉES PRÊTES                 ║
╠══════════════════════════════════════════════════════════════════════╣
║  Observations  : 12 000 → Train : 9 600  |  Test : 2 400            ║
║  Features      : {X_train_sc.shape[1]} (10 brutes + 4 OHE - 1 ref)               ║
║  Taux défaut   : Train {y_train.mean()*100:.2f}%  |  Test {y_test.mean()*100:.2f}%               ║
║  Valeurs manq. : 0 (toutes imputées)                                 ║
║  Outliers      : cappés au P99 (income, dti, utilization_rate)       ║
║  Encodage      : OHE employment_status (ref = Permanent)             ║
║  Scaling       : StandardScaler fitté sur train uniquement           ║
║                                                                      ║
║  PROCHAINE ÉTAPE : train.py — Entraînement Régression Logistique     ║
╚══════════════════════════════════════════════════════════════════════╝
""")
