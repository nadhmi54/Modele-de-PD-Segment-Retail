# Modèle de Probabilité de Défaut (PD) — Segment Retail

> Projet réalisé dans le cadre d'un stage chez **EY**  
> Encadrant : Dhirar  

---

## Présentation du Projet

Ce projet construit un modèle de **Probabilité de Défaut (PD)** pour le segment retail bancaire. L'objectif est d'estimer, à partir du profil financier et comportemental d'un client, la probabilité qu'il tombe en défaut de paiement dans les **12 prochains mois**.

Le projet couvre l'ensemble du cycle de vie d'un modèle de risque de crédit : compréhension des données, analyse exploratoire, prétraitement, modélisation, évaluation des performances et déploiement en production via une API REST et une interface web.

---

## Architecture du Projet

```
PD_Model/
│
├── data/                          # Données brutes
│   ├── simulated_retail_pd_dataset_10_variables.xlsx
│   └── retail_pd_10_variables_data_dictionary.xlsx
│
├── src/                           # Scripts de traitement
│   ├── run_eda.py                 # Analyse exploratoire
│   ├── preprocessing.py           # Pipeline de prétraitement
│   ├── train.py                   # Entraînement + MLflow tracking
│   ├── evaluate.py                # Évaluation des performances
│   ├── predict.py                 # Pipeline d'inférence
│   └── utils.py                   # Fonctions utilitaires
├──app
├── api/                           # Backend REST
│   ├── app.py                     # Serveur FastAPI
│   └── schemas.py                 # Schémas de validation Pydantic
│
├── frontend/                      # Interface utilisateur
│   └── app.py                     # Application Streamlit
│
├── models/                        # Artefacts sauvegardés
│   ├── logistic_regression.pkl
│   ├── scaler.pkl
│   ├── imputation_medians.pkl
│   ├── capping_caps.pkl
│   ├── train_test_data.pkl
│   └── evaluation_results.pkl
│
├── notebooks/
│   └── PD_Model.ipynb             # Notebook complet (EDA → Évaluation)
│
├── docs/                          # Rapports Word par partie
│   ├── Partie1_Comprehension_Variables.docx
│   ├── Rapport_EDA_Preprocessing_PD_Model.docx
│   ├── Partie3_Construction_Modele.docx
│   └── Partie4_Evaluation_Performances.docx
│
├── mlruns/                        # Expériences MLflow (SQLite)
│   └── mlflow.db
│
├── reports/figures/               # Visualisations générées
│
├── config.yaml                    # Configuration centralisée
├── requirements.txt               # Dépendances Python
├── Dockerfile                     # Image Docker
└── docker-entrypoint.sh           # Script de démarrage des services
```

---

## Technologies Utilisées

### Machine Learning & Data
| Outil | Rôle |
|-------|------|
| **Python 3.12** | Langage principal |
| **pandas / numpy** | Manipulation des données |
| **scikit-learn** | Modélisation et prétraitement |
| **scipy** | Tests statistiques |
| **openpyxl** | Lecture des fichiers Excel |
| **joblib** | Sérialisation des modèles |

### Visualisation
| Outil | Rôle |
|-------|------|
| **matplotlib / seaborn** | Visualisations EDA et évaluation |
| **plotly** | Graphiques interactifs (frontend) |

### API & Backend
| Outil | Rôle |
|-------|------|
| **FastAPI** | Framework REST API |
| **uvicorn** | Serveur ASGI |
| **Pydantic v2** | Validation des données entrantes |

### Frontend
| Outil | Rôle |
|-------|------|
| **Streamlit** | Interface de scoring |

### MLOps
| Outil | Rôle |
|-------|------|
| **MLflow** | Tracking des expériences (SQLite backend) |
| **Docker** | Conteneurisation et déploiement |

### Configuration
| Outil | Rôle |
|-------|------|
| **PyYAML** | Fichier de configuration centralisé |

---

## Étapes du Projet

### Partie 1 — Compréhension des Variables
- Analyse du dictionnaire de données
- Identification des 10 variables : 8 numériques, 1 binaire, 1 catégorielle
- Compréhension métier de chaque variable dans le contexte du risque de crédit

**Variables du modèle :**
| Variable | Type | Description |
|----------|------|-------------|
| `bureau_score` | Numérique | Score de crédit externe (300–850) |
| `monthly_income` | Numérique | Revenu mensuel net |
| `dti` | Numérique | Ratio dette / revenu |
| `utilization_rate` | Numérique | Taux d'utilisation du crédit |
| `num_credit_lines` | Numérique | Nombre de lignes de crédit actives |
| `missed_payments_3m` | Numérique | Paiements manqués sur 3 mois |
| `overdrawn_days_3m` | Numérique | Jours à découvert sur 3 mois |
| `max_dpd_12m` | Numérique | Maximum de jours de retard sur 12 mois |
| `salary_domiciliation` | Binaire | Domiciliation du salaire (0/1) |
| `employment_status` | Catégorielle | Statut d'emploi (5 modalités) |

---

### Partie 2 — Analyse Exploratoire & Prétraitement

**Analyse Exploratoire (EDA) :**
- Distribution des variables et détection des valeurs aberrantes
- Analyse de la variable cible `default_12m` : taux de défaut ~17%
- Matrice de corrélation et analyse bivariée
- Visualisation des distributions par statut de défaut

**Pipeline de Prétraitement (sans feature engineering) :**

```
Données brutes
    │
    ▼
1. Imputation par la médiane
   (bureau_score, monthly_income — valeurs manquantes)
    │
    ▼
2. Écrêtage au P99
   (monthly_income, dti, utilization_rate — valeurs extrêmes)
    │
    ▼
3. One-Hot Encoding
   (employment_status — référence : Permanent)
    │
    ▼
4. StandardScaler
   (variables numériques uniquement — fit sur train seulement)
    │
    ▼
5. Split Train / Test  (80% / 20% — stratifié)
```

> Le scaler est fitté **uniquement sur le train** pour éviter toute fuite de données vers le test.

---

### Partie 3 — Construction du Modèle

**Algorithme choisi :** Régression Logistique

**Justification :**
- Interprétabilité des coefficients (exigence Bâle II)
- Robustesse sur les données tabulaires déséquilibrées
- Probabilités directement calibrées

**Hyperparamètres :**
```
C             = 1.0
class_weight  = balanced    ← gestion du déséquilibre (17% défauts)
solver        = lbfgs
max_iter      = 1000
```

**Validation croisée :** 5-fold stratifiée

**Résultats CV :**
- AUC-ROC moyen : **0.8543** ± 0.0066
- F1-Score moyen : **0.5004** ± 0.0130

**Tracking MLflow :** chaque run est enregistré dans `mlruns/mlflow.db` avec les paramètres, métriques et artefacts du modèle.

---

### Partie 4 — Évaluation des Performances

**Résultats sur le jeu de test :**

| Métrique | Valeur | Benchmark Bâle II |
|----------|--------|-------------------|
| **AUC-ROC** | **0.8625** | ≥ 0.75 ✅ |
| **Gini** | **0.7251** | ≥ 0.50 ✅ |
| **KS Statistic** | **0.5775** | ≥ 0.30 ✅ |
| F1-Score | 0.5042 | — |
| Recall (défauts) | 75.08% | — |
| Precision | 37.95% | — |
| Spécificité | 81.05% | — |

**Matrice de confusion (seuil = 0.47) :**
```
              Prédit Non-défaut    Prédit Défaut
Réel Non-défaut       1685              394
Réel Défaut            80               241
```

**Grille de risque :**
| Classe | Seuil PD | Décision |
|--------|----------|----------|
| Faible | < 15% | Approuver |
| Modéré | 15–35% | Approuver avec suivi |
| Élevé | 35–55% | Revue manuelle |
| Très élevé | > 55% | Refuser |

---

### Partie 5 — Déploiement

**Architecture de déploiement :**

```
┌─────────────────────────────────────────────┐
│                  Docker                      │
│                                             │
│  ┌──────────────┐     ┌──────────────────┐  │
│  │  Streamlit   │────▶│   FastAPI        │  │
│  │  :8501       │HTTP │   :8000          │  │
│  └──────────────┘     └────────┬─────────┘  │
│                                │             │
│                       ┌────────▼─────────┐  │
│                       │  predict.py      │  │
│                       │  Pipeline ML     │  │
│                       └────────┬─────────┘  │
│                                │             │
│                       ┌────────▼─────────┐  │
│                       │  models/*.pkl    │  │
│                       └──────────────────┘  │
└─────────────────────────────────────────────┘
```

**Endpoints API :**
| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | Statut de l'API |
| GET | `/model-info` | Métriques du modèle |
| POST | `/predict` | Score PD d'un client |
| GET | `/docs` | Documentation Swagger |

**Interface Streamlit :**
- Formulaire de saisie du profil client
- Jauge de score PD
- Classe de risque et recommandation de décision
- Graphique des facteurs contributifs (rouge = risque, vert = protection)

---

## Lancement du Projet

### Option 1 — Local

```bash
# Depuis le dossier PD_Model/

# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Lancer le pipeline complet
python -X utf8 src/preprocessing.py
python -X utf8 src/train.py
python -X utf8 src/evaluate.py

# 3. Lancer l'API (Terminal 1)
python -X utf8 -m uvicorn app.api.app:app --port 8000 --reload 

# 4. Lancer le Frontend (Terminal 2)
python -X utf8 -m streamlit run app/frontend/app.py --server.port 8502
```

### Option 2 — Docker

```bash
# Depuis le dossier PD_Model/

# Builder l'image
docker build -t pd-scoring:1.0 .

# Lancer le conteneur
docker run -d --name pd-scoring-app -p 8000:8000 -p 8501:8501 pd-scoring:1.0

# Voir les logs
docker logs pd-scoring-app
```

### Accès aux services

| Service | URL |
|---------|-----|
| Frontend Streamlit | http://localhost:8501 |
| API FastAPI | http://localhost:8000 |
| Documentation API | http://localhost:8000/docs |
| MLflow UI | `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db` |

---

## Résultats Clés

Le modèle dépasse tous les benchmarks avec un **AUC-ROC de 0.8625** et un **Gini de 0.7251**, ce qui le classe dans la catégorie des modèles de bonne qualité discriminante pour le risque de crédit retail.

Le recall de **75%** sur la classe défaut garantit que 3 défauts sur 4 sont correctement identifiés, minimisant ainsi le risque de faux négatifs — priorité absolue dans un contexte de gestion du risque bancaire.

---

*EY Advisory — Modélisation Risque de Crédit Retail*
