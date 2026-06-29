# PD Scoring — Segment Retail

> **Projet de stage — EY Advisory**  
> Modélisation de la Probabilité de Défaut (PD) · Régression Logistique 

---

## Vue d'ensemble

Ce projet construit un modèle complet de **Probabilité de Défaut (PD)** pour le segment retail bancaire, conforme aux exigences réglementaires **Bâle II / III**.

À partir du profil financier et comportemental d'un client, le modèle estime la probabilité qu'il tombe en défaut de paiement dans les **12 prochains mois**.

Le projet couvre l'intégralité du cycle de vie d'un modèle de risque de crédit :

```
Données brutes  →  EDA  →  Prétraitement  →  Discrétisation WOE  →  Modèle  →  API + Interface
```

---

## Performances du Modèle (après WOE)

| Métrique | Valeur | Benchmark Bâle II | Statut |
|---|---|---|---|
| **AUC-ROC** | **0.8629** | ≥ 0.75 | ✅ Excellent |
| **Gini** | **0.7257** | ≥ 0.50 | ✅ Excellent |
| **KS Statistic** | **0.6115** | ≥ 0.30 | ✅ Excellent |
| **Rappel (défauts)** | **76.0%** | — | ✅ |
| **F1-Score** | **0.5584** | — | ✅ |

---

## Architecture du Projet

```
PD_Model/
│
├── data/                          # Données brutes
│   ├── simulated_retail_pd_dataset_10_variables.xlsx
│   └── retail_pd_10_variables_data_dictionary.xlsx
│
├── notebooks/
│   └── PD_Model.ipynb             # Notebook complet (Parties 1 → 5)
│
├── src/                           # Scripts Python
│   ├── preprocessing.py           # Pipeline de prétraitement
│   ├── train.py                   # Entraînement + MLflow tracking
│   ├── evaluate.py                # Évaluation des performances
│   ├── predict.py                 # Pipeline d'inférence (WOE)
│   ├── run_eda.py                 # Analyse exploratoire
│   ├── add_woe_section.py         # Générateur Partie 5 WOE
│   └── utils.py                   # Fonctions utilitaires
│
├── app/
│   ├── api/
│   │   ├── app.py                 # Serveur FastAPI
│   │   └── schemas.py             # Schémas Pydantic
│   └── frontend/
│       └── app.py                 # Interface Streamlit
│
├── models/                        # Artefacts sauvegardés
│   ├── logistic_regression_woe.pkl   # Modèle final (WOE)
│   ├── woe_maps.pkl                  # Tables WOE par variable
│   ├── woe_bins.pkl                  # Coupures de discrétisation
│   ├── imputation_medians.pkl
│   ├── capping_caps.pkl
│   └── evaluation_results.pkl
│
├── docs/                          # Documentation Word
├── reports/figures/               # Visualisations générées
├── mlruns/                        # Expériences MLflow
├── config.yaml                    # Configuration centralisée
├── requirements.txt
├── Dockerfile
└── docker-entrypoint.sh
```

---

## Étapes du Projet

### Partie 1 — Compréhension des Variables

Analyse du dictionnaire de données et compréhension métier des **10 variables** du modèle :

| Variable | Type | Description |
|---|---|---|
| `bureau_score` | Numérique | Score de crédit externe (300–850) |
| `monthly_income` | Numérique | Revenu mensuel net |
| `dti` | Numérique | Ratio dette / revenu |
| `utilization_rate` | Numérique | Taux d'utilisation des lignes de crédit |
| `missed_payments_3m` | Numérique | Paiements manqués sur 3 mois |
| `overdrawn_days_3m` | Numérique | Jours à découvert sur 3 mois |
| `max_dpd_12m` | Numérique | Maximum de jours de retard sur 12 mois |
| `number_of_credit_inquiries_6m` | Numérique | Demandes de crédit sur 6 mois |
| `salary_domiciliation` | Binaire | Domiciliation du salaire (0/1) |
| `employment_status` | Catégorielle | Statut d'emploi (5 modalités) |

---

### Partie 2 — Analyse Exploratoire (EDA) & Prétraitement

**EDA :**
- Taux de défaut global : **13.37%** (déséquilibre 1:6.5)
- Détection des valeurs aberrantes et des valeurs manquantes
- Matrice de corrélation et analyse bivariée par tranches
- Visualisation des distributions par statut de défaut (défaut vs non-défaut)

**Pipeline de prétraitement :**

```
Données brutes
    │
    ▼  1. Imputation par la médiane  (bureau_score, monthly_income)
    │
    ▼  2. Écrêtage au P99            (monthly_income, dti, utilization_rate)
    │
    ▼  3. One-Hot Encoding           (employment_status — référence : Permanent)
    │
    ▼  4. Split Train / Test         (80% / 20% — stratifié sur la cible)
    │
    ▼  5. StandardScaler             (fit sur train uniquement — anti data leakage)
```

---

### Partie 3 — Construction du Modèle

**Algorithme :** Régression Logistique — standard industriel Bâle II

**Hyperparamètres :**
```
C             = 1.0      (régularisation L2)
class_weight  = None     (déséquilibre géré via le seuil ROC)
solver        = lbfgs
max_iter      = 1000
```

**Équation du modèle :**
```
log(p / (1-p)) = B0 + B1·WOE_X1 + B2·WOE_X2 + ...
```

**Seuil de décision :** calculé via la courbe ROC (maximise TPR − FPR) — remplace le seuil arbitraire de 0.5.

**Validation croisée 5-fold stratifiée :**
- AUC-ROC moyen : **0.8543** ± 0.0066

---

### Partie 4 — Évaluation des Performances

**Matrice de confusion (seuil optimal ROC) :**
```
                   Prédit Non-défaut    Prédit Défaut
Réel Non-défaut          1689               390
Réel Défaut                75               246
```

**Grille de décision :**
| Classe de risque | Seuil PD | Décision |
|---|---|---|
| Faible | < 15% | Approuver |
| Modéré | 15 – 35% | Approuver avec suivi |
| Élevé | 35 – 55% | Revue manuelle |
| Très élevé | > 55% | Refuser |

---

### Partie 5 — Discrétisation & Weight of Evidence (WOE)

Transformation de chaque variable continue en **classes monotones** avec calcul du **WOE** — méthode standard Bâle II.

**Principe :**
```
Valeur brute  →  Classe (max 4)  →  WOE  →  Régression logistique
```

**Coupures par variable :**

| Variable | Sens | Classe 1 | Classe 2 | Classe 3 | Classe 4 |
|---|---|---|---|---|---|
| `bureau_score` | Décroissant | 300–500 | 500–600 | 600–680 | 680–850 |
| `max_dpd_12m` | Croissant | 0 j | 1–15 j | 15–45 j | 45+ j |
| `dti` | Croissant | 0–30% | 30–50% | 50–70% | 70%+ |
| `utilization_rate` | Croissant | 0–30% | 30–50% | 50–70% | 70%+ |
| `missed_payments_3m` | Croissant | 0 | 1 | 2 | 3+ |
| `overdrawn_days_3m` | Croissant | 0 j | 1–5 j | 5–15 j | 15+ j |
| `number_of_credit_inquiries_6m` | Croissant | 0–1 | 1–3 | 3–5 | 5+ |
| `monthly_income` | Décroissant | 0–1500 | 1500–2500 | 2500–4000 | 4000+ |

**Apport du WOE :**
- Probabilités **bien calibrées** (pas de distorsion)
- Relation **monotone et linéaire** entre chaque variable et le défaut
- Modèle **plus stable** et **conforme Bâle II**
- Amélioration du KS : 0.5775 → **0.6115**

---

### Partie 6 — Déploiement (API + Interface)

**Architecture :**

```
┌─────────────────────────────────────────┐
│                 Docker                   │
│                                         │
│  ┌─────────────┐      ┌──────────────┐  │
│  │  Streamlit  │─────▶│   FastAPI    │  │
│  │   :8502     │ HTTP │    :8000     │  │
│  └─────────────┘      └──────┬───────┘  │
│                               │          │
│                      ┌────────▼───────┐  │
│                      │  predict.py    │  │
│                      │  Pipeline WOE  │  │
│                      └────────┬───────┘  │
│                               │          │
│                      ┌────────▼───────┐  │
│                      │  models/*.pkl  │  │
│                      └────────────────┘  │
└─────────────────────────────────────────┘
```

**Endpoints API :**
| Méthode | Endpoint | Description |
|---|---|---|
| GET | `/health` | Statut de l'API |
| GET | `/model-info` | Métriques du modèle |
| POST | `/predict` | Score PD d'un client |
| GET | `/docs` | Documentation Swagger |

**Interface Streamlit :**
- Formulaire de saisie du profil client (10 variables)
- Jauge de score PD (0 → 100%)
- Classe de risque et recommandation de décision
- Graphique des facteurs contributifs au score

---

## Technologies Utilisées

| Catégorie | Outils |
|---|---|
| **Langage** | Python 3.12 |
| **Data & ML** | pandas, numpy, scikit-learn, scipy |
| **Visualisation** | matplotlib, seaborn, plotly |
| **API Backend** | FastAPI, uvicorn, Pydantic v2 |
| **Frontend** | Streamlit |
| **MLOps** | MLflow (SQLite backend) |
| **Déploiement** | Docker |
| **Configuration** | PyYAML |

---

## Lancement du Projet

### Option 1 — Local

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Lancer l'API (Terminal 1)
python -X utf8 -m uvicorn app.api.app:app --port 8000 --reload

# 3. Lancer le Frontend (Terminal 2)
python -X utf8 -m streamlit run app/frontend/app.py --server.port 8502
```

### Option 2 — Docker

```bash
# Builder et lancer
docker build -t pd-scoring:1.0 .
docker run -d --name pd-scoring-app -p 8000:8000 -p 8502:8502 pd-scoring:1.0
```

### Accès aux services

| Service | URL |
|---|---|
| Interface Streamlit | http://localhost:8502 |
| API FastAPI | http://localhost:8000 |
| Documentation Swagger | http://localhost:8000/docs |
| MLflow UI | `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db` |

---

## Résultats Clés

Le modèle final (après discrétisation WOE) dépasse tous les benchmarks réglementaires :

- **AUC-ROC = 0.8629** — le modèle distingue correctement 86% des paires (défaut / non-défaut)
- **Gini = 0.7257** — indicateur clé des régulateurs Bâle, au-dessus du seuil excellent (> 0.60)
- **KS = 0.6115** — séparation maximale entre les distributions de défaut et non-défaut
- **76% des défauts détectés** — 3 clients risqués sur 4 sont correctement identifiés

---

*EY Advisory — Modélisation Risque de Crédit Retail · Stage 2025*
