# -*- coding: utf-8 -*-
"""
Frontend Streamlit — Scoring PD Retail
Communique avec l'API FastAPI via HTTP
"""

import sys, os, yaml, requests
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

API_URL = cfg['frontend']['api_url']

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PD Scoring | EY Advisory",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #F0F4F8; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

.header-banner {
  background: linear-gradient(135deg, #0F1B2D 0%, #1A2E4A 50%, #2E5B8C 100%);
  border-radius: 18px; padding: 30px 40px; margin-bottom: 28px;
  display: flex; align-items: center; justify-content: space-between;
  box-shadow: 0 8px 32px rgba(15,27,45,0.35);
}
.header-left h1 { color: #FFFFFF; font-size: 2rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
.header-left p  { color: #90A8C3; font-size: 0.9rem; margin: 6px 0 0; }
.header-right { text-align: right; }
.ey-badge {
  background: #F0A500; color: #0F1B2D; border-radius: 8px;
  padding: 7px 18px; font-weight: 800; font-size: 0.9rem; display: inline-block;
}
.api-status { margin-top: 8px; font-size: 0.8rem; }

.card {
  background: #FFFFFF; border-radius: 16px; padding: 26px;
  box-shadow: 0 2px 16px rgba(0,0,0,0.06); border: 1px solid #E2EBF5;
  margin-bottom: 20px;
}
.card-title {
  font-size: 0.78rem; font-weight: 700; color: #64748B;
  text-transform: uppercase; letter-spacing: 0.1em;
  margin-bottom: 18px; padding-bottom: 12px;
  border-bottom: 3px solid #F0A500;
}
.section-header {
  font-size: 0.88rem; font-weight: 600; color: #1A2E4A;
  margin: 18px 0 12px; display: flex; align-items: center; gap: 8px;
}
.divider { height: 1px; background: #E8EDF5; margin: 20px 0; }

.score-big {
  font-size: 4.5rem; font-weight: 900; letter-spacing: -3px;
  background: linear-gradient(135deg, #2E5B8C, #1A2E4A);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  line-height: 1;
}
.score-unit { font-size: 1.5rem; font-weight: 600; color: #64748B; }

.risk-pill {
  display: inline-block; border-radius: 50px;
  padding: 10px 28px; font-weight: 700; font-size: 1.05rem;
  margin: 10px 0; letter-spacing: 0.02em;
}
.pill-low    { background:#DCFCE7; color:#15803D; border: 2px solid #22C55E; }
.pill-mod    { background:#FEF9C3; color:#92400E; border: 2px solid #EAB308; }
.pill-high   { background:#FFEDD5; color:#C2410C; border: 2px solid #F97316; }
.pill-vhigh  { background:#FEE2E2; color:#991B1B; border: 2px solid #EF4444; }

.reco-box {
  border-radius: 12px; padding: 16px 22px; font-weight: 600;
  font-size: 0.95rem; text-align: center; margin-top: 14px;
}
.reco-approve { background:#F0FDF4; color:#15803D; border: 2px solid #22C55E; }
.reco-watch   { background:#FFFBEB; color:#92400E; border: 2px solid #EAB308; }
.reco-review  { background:#FFF7ED; color:#C2410C; border: 2px solid #F97316; }
.reco-reject  { background:#FEF2F2; color:#991B1B; border: 2px solid #EF4444; }

.metric-mini {
  background: #F8FAFC; border-radius: 10px; padding: 14px;
  text-align: center; border: 1px solid #E2EBF5;
}
.metric-mini .val { font-size: 1.4rem; font-weight: 700; color: #2E5B8C; }
.metric-mini .lbl { font-size: 0.7rem; color: #64748B; margin-top: 3px; font-weight: 500; }

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0F1B2D 0%, #1A2E4A 100%);
}
[data-testid="stSidebar"] * { color: #CBD5E0 !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #FFFFFF !important; }

.stButton > button {
  background: linear-gradient(135deg, #2E5B8C 0%, #1A2E4A 100%) !important;
  color: white !important; border: none !important; border-radius: 12px !important;
  padding: 14px 32px !important; font-weight: 700 !important; font-size: 1rem !important;
  width: 100% !important; letter-spacing: 0.02em !important;
  box-shadow: 0 4px 16px rgba(46,91,140,0.35) !important;
  transition: all 0.2s ease !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 24px rgba(46,91,140,0.45) !important;
}

.empty-state {
  text-align: center; padding: 50px 30px;
}
.empty-state .icon { font-size: 5rem; margin-bottom: 16px; }
.empty-state h3 { color: #2E5B8C; font-size: 1.3rem; font-weight: 700; margin-bottom: 8px; }
.empty-state p  { color: #64748B; font-size: 0.9rem; line-height: 1.7; }
.perf-pills { display: flex; justify-content: center; gap: 16px; margin-top: 24px; flex-wrap: wrap; }
.perf-pill  {
  background: #EBF5FF; color: #2E5B8C; border-radius: 50px;
  padding: 8px 20px; font-weight: 700; font-size: 0.88rem;
  border: 1px solid #BFDBFE;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except:
        return False


def call_predict(payload):
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=8)
        if r.status_code == 200:
            return r.json(), None
        err = r.json()
        return None, err.get('detail', [str(err)])
    except requests.exceptions.ConnectionError:
        return None, [f"API non joignable sur {API_URL}. Lancez : uvicorn api.app:app --port 8000"]
    except Exception as e:
        return None, [str(e)]


def gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        number={'suffix': '%', 'font': {'size': 40, 'color': '#1A2E4A', 'family': 'Inter'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#94A3B8',
                     'tickfont': {'size': 10}, 'nticks': 6},
            'bar': {'color': '#2E5B8C', 'thickness': 0.25},
            'bgcolor': '#F8FAFC',
            'borderwidth': 0,
            'steps': [
                {'range': [0,  15], 'color': '#DCFCE7'},
                {'range': [15, 35], 'color': '#FEF9C3'},
                {'range': [35, 55], 'color': '#FFEDD5'},
                {'range': [55,100], 'color': '#FEE2E2'},
            ],
            'threshold': {
                'line': {'color': '#1A2E4A', 'width': 3},
                'thickness': 0.8,
                'value': round(score * 100, 1)
            }
        }
    ))
    fig.update_layout(
        height=240, margin=dict(t=16, b=8, l=24, r=24),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter'}
    )
    return fig


def drivers_chart(drivers):
    items  = sorted(drivers.items(), key=lambda x: abs(x[1]), reverse=True)[:7]
    labels = [k.replace('employment_status_', 'emp_').replace('_', ' ') for k, _ in items]
    values = [v for _, v in items]
    colors = ['#EF4444' if v > 0 else '#22C55E' for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation='h',
        marker={'color': colors, 'line': {'width': 0}},
        text=[f"{v:+.3f}" for v in values],
        textposition='outside',
        textfont={'size': 10.5, 'family': 'Inter', 'color': '#1A2E4A'}
    ))
    fig.update_layout(
        height=300, margin=dict(t=8, b=8, l=8, r=55),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#F1F5F9',
                   zeroline=True, zerolinecolor='#94A3B8', zerolinewidth=1.5),
        yaxis=dict(autorange='reversed', tickfont={'size': 11}),
        font={'family': 'Inter'},
    )
    return fig


# ── HEADER ────────────────────────────────────────────────────────────────────
api_ok    = api_health()
api_dot   = "🟢" if api_ok else "🔴"
api_label = "API connectée" if api_ok else "API hors ligne"

st.markdown(f"""
<div class="header-banner">
  <div class="header-left">
    <h1>📊 PD Scoring — Retail</h1>
    <p>Probabilité de Défaut · Régression Logistique · Bâle II/III</p>
  </div>
  <div class="header-right">
    <div class="ey-badge">EY Advisory</div>
    <div class="api-status" style="color:#90A8C3;">{api_dot} {api_label}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 PD Model")
    st.markdown("---")

    st.markdown("### 👤 Informations Dossier")
    client_name = st.text_input("Nom du client", placeholder="Ex: Ahmed Ben Ali",
                                label_visibility="visible")
    client_ref  = st.text_input("Référence dossier", placeholder="Ex: REF-2026-001")

    st.markdown("---")
    st.markdown("### 📈 Performance Modèle")
    metrics = [("AUC-ROC","0.8625"), ("Gini","0.7251"), ("KS","0.5775"), ("Rappel","75.1%")]
    c1, c2 = st.columns(2)
    for i, (lbl, val) in enumerate(metrics):
        (c1 if i % 2 == 0 else c2).metric(lbl, val)

    st.markdown("---")
    st.markdown("### 🎯 Grille de Risque")
    risk_grid = [
        ("🟢","Faible",    "PD < 15%",   "Approuver"),
        ("🟡","Modéré",   "15–35%",     "Avec suivi"),
        ("🟠","Élevé",    "35–55%",     "Revue"),
        ("🔴","Très élevé","> 55%",      "Refuser"),
    ]
    for icon, label, range_, reco in risk_grid:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;
                    padding:6px 0;border-bottom:1px solid #2E3F56;">
          <span>{icon}</span>
          <div>
            <div style="font-weight:600;font-size:0.82rem;">{label} · {range_}</div>
            <div style="font-size:0.75rem;opacity:0.7;">{reco}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ── MAIN COLUMNS ──────────────────────────────────────────────────────────────
form_col, res_col = st.columns([1.15, 0.85], gap="large")

# ═══════════════════════════════════════════════
# FORMULAIRE
# ═══════════════════════════════════════════════
with form_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🧾 Profil Client</div>', unsafe_allow_html=True)

    # Section 1
    st.markdown('<div class="section-header">💰 Situation Financière</div>',
                unsafe_allow_html=True)
    f1, f2 = st.columns(2)
    with f1:
        monthly_income = st.number_input(
            "Revenu mensuel net (DT)",
            min_value=0.0, max_value=50000.0, value=2000.0, step=100.0,
            help="Revenu mensuel après impôts en dinars tunisiens")
        dti = st.number_input(
            "Ratio dette/revenu (DTI)",
            min_value=0.0, max_value=5.0, value=0.35, step=0.01, format="%.2f",
            help="Mensualités totales / revenu mensuel. Ex: 0.35 = 35%")
    with f2:
        bureau_score = st.number_input(
            "Score bureau crédit",
            min_value=300, max_value=850, value=650, step=5,
            help="Score de crédit externe. 300 = très risqué · 850 = excellent")
        utilization_rate = st.number_input(
            "Taux utilisation crédit",
            min_value=0.0, max_value=2.0, value=0.40, step=0.01, format="%.2f",
            help="Encours utilisé / plafond autorisé total")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Section 2
    st.markdown('<div class="section-header">⚠️ Comportement de Paiement (3 mois)</div>',
                unsafe_allow_html=True)
    f3, f4, f5 = st.columns(3)
    with f3:
        missed_payments = st.number_input(
            "Paiements manqués", min_value=0, max_value=30, value=0,
            help="Nombre de paiements en retard sur les 3 derniers mois")
    with f4:
        overdrawn_days = st.number_input(
            "Jours à découvert", min_value=0, max_value=90, value=0,
            help="Jours avec solde négatif sur 3 mois")
    with f5:
        max_dpd = st.number_input(
            "Max DPD 12 mois", min_value=0, max_value=365, value=0,
            help="Nombre maximum de jours de retard sur 12 mois")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Section 3
    st.markdown('<div class="section-header">👔 Emploi & Bancarisation</div>',
                unsafe_allow_html=True)
    f6, f7, f8 = st.columns(3)
    with f6:
        employment_status = st.selectbox(
            "Statut d'emploi",
            options=['Permanent', 'Self_Employed', 'Student', 'Temporary', 'Unemployed'],
            format_func=lambda x: {
                'Permanent':    'Permanent (CDI)',
                'Self_Employed':'Indépendant',
                'Student':      'Étudiant',
                'Temporary':    'Temporaire (CDD)',
                'Unemployed':   'Sans emploi'
            }.get(x, x)
        )
    with f7:
        num_credit_lines = st.number_input(
            "Lignes de crédit", min_value=0, max_value=50, value=3)
    with f8:
        salary_domiciliation = st.selectbox(
            "Salaire domicilié",
            options=[1, 0],
            format_func=lambda x: 'Oui' if x == 1 else 'Non'
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    score_btn = st.button("⚡  Calculer la Probabilité de Défaut",
                          use_container_width=True)


# ═══════════════════════════════════════════════
# RÉSULTATS
# ═══════════════════════════════════════════════
with res_col:
    if not score_btn:
        st.markdown("""
        <div class="card">
          <div class="empty-state">
            <div class="icon">🎯</div>
            <h3>Prêt à scorer un client</h3>
            <p>
              Renseignez le profil client dans le formulaire<br>
              puis cliquez sur <strong>Calculer la PD</strong>
            </p>
            <div class="perf-pills">
              <div class="perf-pill">AUC 0.8625</div>
              <div class="perf-pill">Gini 0.7251</div>
              <div class="perf-pill">KS 0.5775</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        payload = {
            'bureau_score':        bureau_score,
            'monthly_income':      monthly_income,
            'dti':                 dti,
            'utilization_rate':    utilization_rate,
            'num_credit_lines':    num_credit_lines,
            'missed_payments_3m':  missed_payments,
            'overdrawn_days_3m':   overdrawn_days,
            'max_dpd_12m':         max_dpd,
            'salary_domiciliation': salary_domiciliation,
            'employment_status':   employment_status,
        }

        with st.spinner("Calcul en cours..."):
            result, errors = call_predict(payload)

        if errors:
            st.error("**Erreur**")
            for e in (errors if isinstance(errors, list) else [errors]):
                st.markdown(f"- {e}")

        else:
            pd_score = result['pd_score']
            pd_pct   = result['pd_pct']
            risk     = result['risk_band']['label']
            reco     = result['risk_band']['recommendation']
            drivers  = result['top_drivers']
            level    = result['risk_band']['level']

            if client_name:
                st.markdown(f"### Résultat — {client_name}")
            if client_ref:
                st.markdown(f"<small style='color:#64748B'>Dossier : {client_ref} · {result['timestamp']}</small>",
                            unsafe_allow_html=True)

            # ── Gauge ─────────────────────────────────────────────
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Score PD</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(gauge(pd_score), use_container_width=True,
                            config={'displayModeBar': False})

            risk_css = {
                'Faible':    'pill-low',
                'Modéré':    'pill-mod',
                'Élevé':     'pill-high',
                'Très élevé':'pill-vhigh',
            }.get(risk, 'pill-vhigh')

            reco_css = {
                'Approuver':           'reco-approve',
                'Approuver avec suivi':'reco-watch',
                'Revue manuelle':      'reco-review',
                'Refuser':             'reco-reject',
            }.get(reco, 'reco-review')

            reco_icons = {
                'Approuver':           '✅',
                'Approuver avec suivi':'👁️',
                'Revue manuelle':      '🔍',
                'Refuser':             '❌',
            }

            st.markdown(f"""
            <div style="text-align:center; padding:4px 0 8px;">
              <span class="risk-pill {risk_css}">Risque {risk}</span>
            </div>
            <div class="reco-box {reco_css}">
              {reco_icons.get(reco,'')} {reco}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # ── Métriques rapides ──────────────────────────────────
            m1, m2, m3 = st.columns(3)
            m1.markdown(f"""<div class="metric-mini">
              <div class="val">{pd_pct:.1f}%</div>
              <div class="lbl">Score PD</div>
            </div>""", unsafe_allow_html=True)
            m2.markdown(f"""<div class="metric-mini">
              <div class="val">{risk}</div>
              <div class="lbl">Classe Risque</div>
            </div>""", unsafe_allow_html=True)
            m3.markdown(f"""<div class="metric-mini">
              <div class="val">{level}/4</div>
              <div class="lbl">Niveau</div>
            </div>""", unsafe_allow_html=True)

            # ── Drivers ───────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Facteurs du Score</div>',
                        unsafe_allow_html=True)
            st.markdown(
                "<small style='color:#64748B'>🔴 Rouge = facteur de risque &nbsp;·&nbsp; "
                "🟢 Vert = facteur protecteur</small>",
                unsafe_allow_html=True
            )
            st.plotly_chart(drivers_chart(drivers), use_container_width=True,
                            config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)
