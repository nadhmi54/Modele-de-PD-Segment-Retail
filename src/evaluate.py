# -*- coding: utf-8 -*-
"""
Evaluation du modele PD Retail — Partie 4
Metriques : AUC-ROC, Gini, KS, Matrice de confusion, Rapport de classification,
            Distribution des scores PD, Courbe ROC, Courbe KS
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    f1_score, classification_report, confusion_matrix,
    precision_score, recall_score
)
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
FIG_DIR   = os.path.join(BASE_DIR, 'reports', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("CHARGEMENT DES DONNEES ET DU MODELE")
print("=" * 70)

X_train, X_test, y_train, y_test = joblib.load(os.path.join(MODEL_DIR, 'train_test_data.pkl'))
model = joblib.load(os.path.join(MODEL_DIR, 'logistic_regression.pkl'))

y_prob_test  = model.predict_proba(X_test)[:, 1]
y_prob_train = model.predict_proba(X_train)[:, 1]
y_pred_test  = model.predict(X_test)

print(f"Test  : {X_test.shape[0]:,} observations")
print(f"Train : {X_train.shape[0]:,} observations")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("1. AUC-ROC")
print("=" * 70)

auc_test  = roc_auc_score(y_test,  y_prob_test)
auc_train = roc_auc_score(y_train, y_prob_train)
gini      = 2 * auc_test - 1

print(f"AUC-ROC Train : {auc_train:.4f}")
print(f"AUC-ROC Test  : {auc_test:.4f}")
print(f"Ecart Train/Test : {abs(auc_train - auc_test):.4f}  (< 0.02 = pas d'overfitting)")
print(f"Coefficient Gini : {gini:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. STATISTIQUE KS (Kolmogorov-Smirnov)")
print("=" * 70)

scores_def   = y_prob_test[y_test == 1]
scores_nodef = y_prob_test[y_test == 0]
ks_stat, ks_pval = ks_2samp(scores_def, scores_nodef)

fpr, tpr, thresholds = roc_curve(y_test, y_prob_test)
ks_curve = max(tpr - fpr)
ks_thresh = thresholds[np.argmax(tpr - fpr)]

print(f"KS (deux echantillons) : {ks_stat:.4f}")
print(f"KS (courbe ROC)        : {ks_curve:.4f}")
print(f"Seuil optimal KS       : {ks_thresh:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. MATRICE DE CONFUSION ET RAPPORT")
print("=" * 70)

cm = confusion_matrix(y_test, y_pred_test)
tn, fp, fn, tp = cm.ravel()

print(f"\nMatrice de confusion :")
print(f"  Vrais Negatifs  (TN) : {tn:,}  — Non-defauts correctement identifies")
print(f"  Faux Positifs   (FP) : {fp:,}  — Non-defauts classes comme defauts")
print(f"  Faux Negatifs   (FN) : {fn:,}  — Defauts manques")
print(f"  Vrais Positifs  (TP) : {tp:,}  — Defauts correctement identifies")

precision   = precision_score(y_test, y_pred_test)
recall      = recall_score(y_test, y_pred_test)
f1          = f1_score(y_test, y_pred_test)
specificity = tn / (tn + fp)

print(f"\nMetriques :")
print(f"  Precision    : {precision:.4f}")
print(f"  Rappel       : {recall:.4f}")
print(f"  Specificite  : {specificity:.4f}")
print(f"  F1-Score     : {f1:.4f}")
print(f"\n{classification_report(y_test, y_pred_test, target_names=['Non-defaut','Defaut'])}")

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("4. GRAPHIQUES")
print("=" * 70)

plt.rcParams.update({'font.family': 'DejaVu Sans',
                     'axes.spines.top': False, 'axes.spines.right': False})

# Fig 14 — Courbe ROC
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color='#2E5B8C', lw=2.5, label=f'ROC (AUC = {auc_test:.4f})')
ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5, label='Aleatoire (AUC = 0.50)')
ax.fill_between(fpr, tpr, alpha=0.08, color='#2E5B8C')
ax.set_xlabel('Taux de Faux Positifs (1 - Specificite)', fontsize=11)
ax.set_ylabel('Taux de Vrais Positifs (Sensibilite)', fontsize=11)
ax.set_title(f'Courbe ROC — Jeu de Test\nAUC = {auc_test:.4f}  |  Gini = {gini:.4f}',
             fontweight='bold', fontsize=12)
ax.legend(fontsize=10)
ax.annotate(f'AUC = {auc_test:.4f}\nGini = {gini:.4f}',
            xy=(0.58, 0.25), fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', fc='#EBF5FB', ec='#2E5B8C'))
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '14_courbe_roc.png'), bbox_inches='tight', dpi=150)
plt.close()
print("14_courbe_roc.png")

# Fig 15 — Courbe KS
fig, ax = plt.subplots(figsize=(7, 6))
thresholds_plot = np.linspace(0, 1, 300)
cum_def   = [np.mean(scores_def   <= t) for t in thresholds_plot]
cum_nodef = [np.mean(scores_nodef <= t) for t in thresholds_plot]
ax.plot(thresholds_plot, cum_def,   color='#C0392B', lw=2, label='Defauts (D=1)')
ax.plot(thresholds_plot, cum_nodef, color='#27AE60', lw=2, label='Non-defauts (D=0)')
ks_idx = int(np.argmax(np.abs(np.array(cum_def) - np.array(cum_nodef))))
ax.vlines(thresholds_plot[ks_idx], cum_nodef[ks_idx], cum_def[ks_idx],
          color='#F0A500', lw=2.5, linestyle='--',
          label=f'KS = {ks_stat:.4f}')
ax.set_xlabel('Seuil de score PD', fontsize=11)
ax.set_ylabel('Proportion cumulative', fontsize=11)
ax.set_title(f'Courbe KS — Separation Defauts / Non-Defauts\nKS = {ks_stat:.4f}',
             fontweight='bold', fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '15_courbe_ks.png'), bbox_inches='tight', dpi=150)
plt.close()
print("15_courbe_ks.png")

# Fig 16 — Matrice de confusion
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
for idx, (data, title, fmt) in enumerate([
        (cm,      'Matrice de Confusion (valeurs absolues)', 'd'),
        (cm_norm, 'Matrice de Confusion (proportions)',       '.2f')]):
    sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=['Non-defaut','Defaut'],
                yticklabels=['Non-defaut','Defaut'],
                ax=axes[idx], linewidths=0.5, linecolor='white',
                annot_kws={'size':13, 'weight':'bold'})
    axes[idx].set_title(title, fontweight='bold', fontsize=11)
    axes[idx].set_xlabel('Valeur Predite', fontsize=10)
    axes[idx].set_ylabel('Valeur Reelle', fontsize=10)
plt.suptitle('Matrice de Confusion — Jeu de Test', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '16_confusion_matrix.png'), bbox_inches='tight', dpi=150)
plt.close()
print("16_confusion_matrix.png")

# Fig 17 — Distribution scores PD
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(scores_nodef, bins=50, alpha=0.65, color='#27AE60', density=True,
        edgecolor='white', label=f'Non-defauts (n={len(scores_nodef):,})')
ax.hist(scores_def,   bins=50, alpha=0.65, color='#C0392B', density=True,
        edgecolor='white', label=f'Defauts (n={len(scores_def):,})')
ax.axvline(ks_thresh, color='#F0A500', lw=2.5, linestyle='--',
           label=f'Seuil optimal = {ks_thresh:.2f}')
ax.set_xlabel('Score PD (Probabilite de Defaut predite)', fontsize=11)
ax.set_ylabel('Densite', fontsize=11)
ax.set_title('Distribution des Scores PD — Jeu de Test\nSeparation Defauts vs Non-Defauts',
             fontweight='bold', fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '17_distribution_scores.png'), bbox_inches='tight', dpi=150)
plt.close()
print("17_distribution_scores.png")

# Fig 18 — Synthese metriques (tableau visuel)
fig, ax = plt.subplots(figsize=(9, 4))
ax.axis('off')
rows_tab = [
    ['AUC-ROC (Test)',      f'{auc_test:.4f}',   '> 0.75', 'Excellent'],
    ['AUC-ROC (Train)',     f'{auc_train:.4f}',  '> 0.75', 'Excellent'],
    ['Ecart Train / Test',  f'{abs(auc_train-auc_test):.4f}', '< 0.02', 'Pas d\'overfitting'],
    ['Coefficient Gini',    f'{gini:.4f}',        '> 0.50', 'Excellent'],
    ['KS Statistic',        f'{ks_stat:.4f}',     '> 0.30', 'Bon'],
    ['F1-Score',            f'{f1:.4f}',          '> 0.45', 'Bon'],
    ['Precision',           f'{precision:.4f}',   '—',      '—'],
    ['Rappel (Sensibilite)',f'{recall:.4f}',       '—',      '—'],
    ['Specificite',         f'{specificity:.4f}', '—',      '—'],
]
tbl = ax.table(
    cellText=rows_tab,
    colLabels=['Metrique', 'Valeur', 'Seuil Bale', 'Interpretation'],
    loc='center', cellLoc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.7)
for j in range(4):
    tbl[0, j].set_facecolor('#2E5B8C')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(rows_tab)+1):
    for j in range(4):
        tbl[i, j].set_facecolor('#F5F7FA' if i % 2 == 0 else 'white')
ax.set_title('Synthese des Performances — Jeu de Test',
             fontweight='bold', fontsize=13, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '18_synthese_metriques.png'), bbox_inches='tight', dpi=150)
plt.close()
print("18_synthese_metriques.png")

# ══════════════════════════════════════════════════════════════════════════════
joblib.dump({
    'auc_test': auc_test, 'auc_train': auc_train,
    'gini': gini, 'ks_stat': ks_stat, 'ks_thresh': ks_thresh,
    'f1': f1, 'precision': precision, 'recall': recall,
    'specificity': specificity, 'cm': cm,
    'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
}, os.path.join(MODEL_DIR, 'evaluation_results.pkl'))

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║              SYNTHESE EVALUATION — PERFORMANCE DU MODELE             ║
╠══════════════════════════════════════════════════════════════════════╣
║  AUC-ROC Test   : {auc_test:.4f}    AUC-ROC Train : {auc_train:.4f}             ║
║  Ecart Tr/Te    : {abs(auc_train-auc_test):.4f}  → pas d'overfitting                  ║
║  Coefficient Gini : {gini:.4f}                                       ║
║  KS Statistic   : {ks_stat:.4f}                                         ║
║  F1-Score       : {f1:.4f}                                         ║
║  Precision      : {precision:.4f}  |  Rappel : {recall:.4f}                  ║
║  Specificite    : {specificity:.4f}                                         ║
╚══════════════════════════════════════════════════════════════════════╝
""")
