import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Style global
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

# Chemins
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'simulated_retail_pd_dataset_10_variables.xlsx')
FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

print('[OK] Configuration des styles et chemins OK\n')

# ── 1. CHARGEMENT ──────────────────────────────────────────────────────
print("== 1. CHARGEMENT DES DONNÉES ==")
df = pd.read_excel(DATA_PATH)
print(f"Dimensions        : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
print(f"Variables         : {list(df.columns)}")
print("\nTypes de variables :")
print(df.dtypes)
print()

# ── 2. VARIABLE CIBLE ──────────────────────────────────────────────────
print("== 2. VARIABLE CIBLE (default_12m) ==")
counts = df['default_12m'].value_counts()
taux   = df['default_12m'].mean() * 100

print(f"Non-défaut (0) : {counts[0]:,}  ({100-taux:.2f}%)")
print(f"Défaut     (1) : {counts[1]:,}  ({taux:.2f}%)")
print(f"[WARNING] Desequilibre des classes : ratio 1:{counts[0]//counts[1]}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(['Non-défaut (0)', 'Défaut (1)'], counts.values,
            color=['#2ecc71', '#e74c3c'], edgecolor='white', width=0.5)
axes[0].set_title('Distribution de la variable cible')
axes[0].set_ylabel('Nombre de clients')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')

axes[1].pie(counts.values, labels=['Non-défaut', 'Défaut'],
            colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%',
            startangle=90, wedgeprops=dict(edgecolor='white'))
axes[1].set_title('Répartition des classes')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '01_target_distribution.png'), bbox_inches='tight')
plt.close()
print("Graphique 01_target_distribution.png généré dans reports/figures/.")
print()

# ── 3. VALEURS MANQUANTES ──────────────────────────────────────────────
print("== 3. VALEURS MANQUANTES ==")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Valeurs manquantes': missing,
    'Pourcentage (%)': missing_pct
}).query('`Valeurs manquantes` > 0')

print("Variables avec valeurs manquantes :")
print(missing_df.to_string())

if not missing_df.empty:
    fig, ax = plt.subplots(figsize=(8, 4))
    vars_missing = missing_df.index.tolist()
    pcts = missing_df['Pourcentage (%)'].values
    bars = ax.barh(vars_missing, pcts, color=['#e67e22', '#3498db'], edgecolor='white')
    ax.set_xlabel('% de valeurs manquantes')
    ax.set_title('Taux de valeurs manquantes par variable')
    ax.axvline(5, color='red', linestyle='--', alpha=0.5, label='Seuil 5%')
    ax.legend()
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{pct}%', va='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '02_missing_values.png'), bbox_inches='tight')
    plt.close()
    print("Graphique 02_missing_values.png généré dans reports/figures/.")

print("\nTaux de défaut selon disponibilité de bureau_score :")
df['bureau_score_missing'] = df['bureau_score'].isnull().astype(int)
print(df.groupby('bureau_score_missing')['default_12m'].mean().rename({0:'Renseigné', 1:'Manquant'}).apply(lambda x: f'{x*100:.2f}%'))

print("\nTaux de défaut selon disponibilité de monthly_income :")
df['monthly_income_missing'] = df['monthly_income'].isnull().astype(int)
print(df.groupby('monthly_income_missing')['default_12m'].mean().rename({0:'Renseigné', 1:'Manquant'}).apply(lambda x: f'{x*100:.2f}%'))
df.drop(columns=['bureau_score_missing','monthly_income_missing'], inplace=True)
print()

# ── 4. STATS DESCRIPTIVES ──────────────────────────────────────────────
print("== 4. STATISTIQUES DESCRIPTIVES ==")
num_cols = ['bureau_score','dti','utilization_rate','max_dpd_12m',
            'missed_payments_3m','overdrawn_days_3m',
            'number_of_credit_inquiries_6m','monthly_income']
desc = df[num_cols].describe().T
desc['skewness'] = df[num_cols].skew().round(3)
desc['kurtosis'] = df[num_cols].kurt().round(3)
print(desc.round(2).to_string())
print()

# ── 5. DISTRIBUTIONS ───────────────────────────────────────────────────
print("== 5. DISTRIBUTIONS DES VARIABLES NUMÉRIQUES (Histo) ==")
fig, axes = plt.subplots(4, 2, figsize=(14, 18))
axes = axes.flatten()
colors = {0: '#2ecc71', 1: '#e74c3c'}
labels = {0: 'Non-défaut', 1: 'Défaut'}
for i, col in enumerate(num_cols):
    ax = axes[i]
    for target in [0, 1]:
        subset = df[df['default_12m'] == target][col].dropna()
        ax.hist(subset, bins=40, alpha=0.6, color=colors[target],
                label=labels[target], edgecolor='white', density=True)
    ax.set_title(f'Distribution : {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Densité')
    ax.legend(fontsize=9)
plt.suptitle('Distributions des variables numériques\n(Défaut vs Non-défaut)', 
             y=1.01, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '03_distributions.png'), bbox_inches='tight')
plt.close()
print("Graphique 03_distributions.png généré dans reports/figures/.")
print()

# ── 6. BOXPLOTS OUTLIERS ───────────────────────────────────────────────
print("== 6. BOXPLOTS & OUTLIERS ==")
fig, axes = plt.subplots(4, 2, figsize=(14, 16))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    ax = axes[i]
    df.boxplot(column=col, by='default_12m', ax=ax,
               boxprops=dict(color='steelblue'),
               medianprops=dict(color='red', linewidth=2),
               whiskerprops=dict(color='steelblue'),
               capprops=dict(color='steelblue'),
               flierprops=dict(marker='o', color='orange', alpha=0.3, markersize=3))
    ax.set_title(col)
    ax.set_xlabel('default_12m (0=Non-défaut, 1=Défaut)')
    ax.set_ylabel(col)
plt.suptitle('Boxplots par statut de défaut\n(outliers en orange)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '04_boxplots.png'), bbox_inches='tight')
plt.close()
print("Graphique 04_boxplots.png généré dans reports/figures/.")

print("\n[INFO] Synthese des outliers potentiels (valeurs > Q3 + 3*IQR) :")
for col in num_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    n_out = (df[col] > q3 + 3*iqr).sum()
    if n_out > 0:
        print(f"  {col:40s} : {n_out} outliers extrêmes")
print()

# ── 7. VARIABLE CATEGORIELLE ───────────────────────────────────────────
print("== 7. VARIABLE CATÉGORIELLE (employment_status) ==")
emp_default = df.groupby('employment_status').agg(
    nb_clients=('default_12m', 'count'),
    taux_defaut=('default_12m', 'mean')
).sort_values('taux_defaut', ascending=False)
emp_default['taux_defaut_pct'] = (emp_default['taux_defaut'] * 100).round(2)
print(emp_default.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
emp_default['taux_defaut_pct'].plot(kind='bar', ax=axes[0],
    color=['#e74c3c','#e67e22','#f39c12','#3498db','#2ecc71'],
    edgecolor='white')
axes[0].set_title("Taux de défaut par statut d'emploi")
axes[0].set_ylabel("Taux de défaut (%)")
axes[0].set_xlabel("")
axes[0].axhline(df['default_12m'].mean()*100, color='black',
                linestyle='--', label=f'Moyenne globale ({df["default_12m"].mean()*100:.1f}%)')
axes[0].legend()
axes[0].tick_params(axis='x', rotation=30)
for i, v in enumerate(emp_default['taux_defaut_pct']):
    axes[0].text(i, v+0.5, f'{v}%', ha='center', fontweight='bold')

emp_default['nb_clients'].plot(kind='bar', ax=axes[1],
    color='steelblue', edgecolor='white')
axes[1].set_title("Nombre de clients par statut d'emploi")
axes[1].set_ylabel("Nombre de clients")
axes[1].set_xlabel("")
axes[1].tick_params(axis='x', rotation=30)
for i, v in enumerate(emp_default['nb_clients']):
    axes[1].text(i, v+30, f'{v:,}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '05_employment_status.png'), bbox_inches='tight')
plt.close()
print("Graphique 05_employment_status.png généré dans reports/figures/.")
print()

# ── 8. CORRÉLATIONS ────────────────────────────────────────────────────
print("== 8. MATRICE DE CORRÉLATION ==")
corr_cols = num_cols + ['salary_domiciliation', 'default_12m']
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdYlGn_r', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, ax=ax,
            cbar_kws={'shrink': 0.8})
ax.set_title('Matrice de corrélation (variables + cible)', fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '06_correlation_matrix.png'), bbox_inches='tight')
plt.close()
print("Graphique 06_correlation_matrix.png généré dans reports/figures/.")

corr_target = df[corr_cols].corr()['default_12m'].drop('default_12m').sort_values()
fig, ax = plt.subplots(figsize=(9, 6))
colors_bar = ['#e74c3c' if v > 0 else '#2ecc71' for v in corr_target.values]
bars = ax.barh(corr_target.index, corr_target.values, color=colors_bar, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Corrélation avec default_12m')
ax.set_title('Corrélation de chaque variable avec la cible', fontweight='bold')
for bar, v in zip(bars, corr_target.values):
    ax.text(v + (0.005 if v >= 0 else -0.005), 
            bar.get_y() + bar.get_height()/2,
            f'{v:.3f}', va='center',
            ha='left' if v >= 0 else 'right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '07_correlation_target.png'), bbox_inches='tight')
plt.close()
print("Graphique 07_correlation_target.png généré dans reports/figures/.")

print("\n[INFO] Interpretation :")
print("  Variables protectrices  (corrélation négative) : bureau_score, monthly_income, salary_domiciliation")
print("  Variables de risque     (corrélation positive)  : max_dpd_12m, utilization_rate, missed_payments_3m")
print()

# ── 9. ANALYSE BIVARIÉE ────────────────────────────────────────────────
print("== 9. ANALYSE BIVARIÉE PAR BUCKETS ==")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# bureau_score
df['score_bucket'] = pd.cut(df['bureau_score'], bins=[300,450,550,620,680,730,780,850],
                             labels=['300-450','450-550','550-620','620-680','680-730','730-780','780-850'])
rate = df.groupby('score_bucket', observed=True)['default_12m'].mean() * 100
axes[0,0].bar(rate.index.astype(str), rate.values, color='steelblue', edgecolor='white')
axes[0,0].set_title('Taux de défaut par tranche de bureau_score')
axes[0,0].set_ylabel('Taux de défaut (%)')
axes[0,0].tick_params(axis='x', rotation=30)
for i, v in enumerate(rate.values):
    axes[0,0].text(i, v+0.2, f'{v:.1f}%', ha='center', fontsize=8)

# dti
df['dti_bucket'] = pd.cut(df['dti'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 10],
                           labels=['0-20%','20-40%','40-60%','60-80%','80-100%','>100%'])
rate2 = df.groupby('dti_bucket', observed=True)['default_12m'].mean() * 100
axes[0,1].bar(rate2.index.astype(str), rate2.values, color='#e67e22', edgecolor='white')
axes[0,1].set_title('Taux de défaut par tranche de DTI')
axes[0,1].set_ylabel('Taux de défaut (%)')
for i, v in enumerate(rate2.values):
    axes[0,1].text(i, v+0.2, f'{v:.1f}%', ha='center', fontsize=8)

# max_dpd_12m
df['dpd_bucket'] = pd.cut(df['max_dpd_12m'], bins=[-1, 0, 15, 30, 60, 90, 1000],
                           labels=['0 jours','1-15','16-30','31-60','61-90','>90'])
rate3 = df.groupby('dpd_bucket', observed=True)['default_12m'].mean() * 100
axes[1,0].bar(rate3.index.astype(str), rate3.values, color='#e74c3c', edgecolor='white')
axes[1,0].set_title('Taux de défaut par tranche de max_dpd_12m')
axes[1,0].set_ylabel('Taux de défaut (%)')
for i, v in enumerate(rate3.values):
    axes[1,0].text(i, v+0.5, f'{v:.1f}%', ha='center', fontsize=8)

# utilization_rate
df['util_bucket'] = pd.cut(df['utilization_rate'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                            labels=['0-20%','20-40%','40-60%','60-80%','80-100%'])
rate4 = df.groupby('util_bucket', observed=True)['default_12m'].mean() * 100
axes[1,1].bar(rate4.index.astype(str), rate4.values, color='#9b59b6', edgecolor='white')
axes[1,1].set_title("Taux de défaut par tranche d'utilization_rate")
axes[1,1].set_ylabel('Taux de défaut (%)')
for i, v in enumerate(rate4.values):
    axes[1,1].text(i, v+0.2, f'{v:.1f}%', ha='center', fontsize=8)

plt.suptitle('Analyse bivariée — Taux de défaut par bucket', 
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '08_bivariate_analysis.png'), bbox_inches='tight')
plt.close()
print("Graphique 08_bivariate_analysis.png généré dans reports/figures/.")
print()

# ── 10. SALARY DOMICILIATION ───────────────────────────────────────────
print("== 10. DOMICILIATION DU SALAIRE ==")
sal = df.groupby('salary_domiciliation')['default_12m'].agg(['mean','count'])
sal.index = ['Non domicilié (0)', 'Domicilié (1)']
sal.columns = ['Taux de défaut', 'Nb clients']
sal['Taux de défaut'] = sal['Taux de défaut'].apply(lambda x: f'{x*100:.2f}%')
print(sal)

fig, ax = plt.subplots(figsize=(7, 4))
vals = df.groupby('salary_domiciliation')['default_12m'].mean() * 100
ax.bar(['Non domicilié (0)', 'Domicilié (1)'], vals.values,
       color=['#e74c3c', '#2ecc71'], edgecolor='white', width=0.4)
ax.set_title('Taux de défaut selon la domiciliation du salaire')
ax.set_ylabel('Taux de défaut (%)')
ax.axhline(df['default_12m'].mean()*100, color='black', linestyle='--',
           label=f'Moyenne ({df["default_12m"].mean()*100:.1f}%)')
ax.legend()
for i, v in enumerate(vals.values):
    ax.text(i, v+0.2, f'{v:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '09_salary_domiciliation.png'), bbox_inches='tight')
plt.close()
print("Graphique 09_salary_domiciliation.png généré dans reports/figures/.")
print()

print("[OK] Fin de l'analyse EDA. Tous les graphes ont ete enregistres dans reports/figures/")
