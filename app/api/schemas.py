# -*- coding: utf-8 -*-
"""
Schémas Pydantic — Validation entrée/sortie de l'API PD
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal, Dict
from enum import Enum


class EmploymentStatus(str, Enum):
    permanent     = "Permanent"
    self_employed = "Self_Employed"
    student       = "Student"
    temporary     = "Temporary"
    unemployed    = "Unemployed"


# ── Entrée ────────────────────────────────────────────────────────────────────
class ClientInput(BaseModel):
    bureau_score:         float = Field(..., ge=300,  le=850,  description="Score bureau crédit (300–850)")
    monthly_income:       float = Field(..., ge=0,    le=50000,description="Revenu mensuel net (DT)")
    dti:                  float = Field(..., ge=0.0,  le=5.0,  description="Ratio dette/revenu")
    utilization_rate:     float = Field(..., ge=0.0,  le=2.0,  description="Taux d'utilisation crédit")
    num_credit_lines:     int   = Field(..., ge=0,    le=50,   description="Nombre de lignes de crédit actives")
    missed_payments_3m:   int   = Field(..., ge=0,    le=30,   description="Paiements manqués sur 3 mois")
    overdrawn_days_3m:    int   = Field(..., ge=0,    le=90,   description="Jours à découvert sur 3 mois")
    max_dpd_12m:          int   = Field(..., ge=0,    le=365,  description="Max jours de retard sur 12 mois")
    salary_domiciliation: int   = Field(..., ge=0,    le=1,    description="Domiciliation salaire (0/1)")
    employment_status:    EmploymentStatus = Field(...,         description="Statut d'emploi")

    model_config = {
        "json_schema_extra": {
            "example": {
                "bureau_score":         650,
                "monthly_income":       2000,
                "dti":                  0.35,
                "utilization_rate":     0.40,
                "num_credit_lines":     3,
                "missed_payments_3m":   0,
                "overdrawn_days_3m":    0,
                "max_dpd_12m":          0,
                "salary_domiciliation": 1,
                "employment_status":    "Permanent"
            }
        }
    }


# ── Sortie ────────────────────────────────────────────────────────────────────
class RiskBand(BaseModel):
    label:          str
    color:          str
    recommendation: str
    level:          int


class PredictionOutput(BaseModel):
    pd_score:        float = Field(..., description="Probabilité de défaut (0–1)")
    pd_pct:          float = Field(..., description="Probabilité de défaut en %")
    risk_band:       RiskBand
    top_drivers:     Dict[str, float] = Field(..., description="Top 5 variables les plus influentes")
    timestamp:       str


class HealthOutput(BaseModel):
    status:    str
    model:     str
    version:   str
    timestamp: str


class ModelInfoOutput(BaseModel):
    algorithm:    str
    C:            float
    class_weight: str
    solver:       str
    n_features:   int
    cv_auc_roc:   float
    test_auc_roc: float
    gini:         float
    ks_statistic: float
