"""
WEEK 3 — Rule-Based Decision Engine
Built on REAL dataset features:
  CreditScore, Income, LoanAmount, YearsExperience, EmploymentType, Education
"""
from dataclasses import dataclass, field
from typing import List

# Hard reject — automatic denial
HARD_REJECT_RULES = [
    ('CreditScore',    lambda v: v < 400,         'Credit score critically low (< 400)'),
    ('EmploymentType', lambda v: v == 'Unemployed','Applicant is unemployed'),
    ('LoanToIncome',   lambda v: v > 2.0,          'Loan amount exceeds 2× annual income'),
]

# Soft negative — score deductions
SOFT_FLAG_RULES = [
    ('CreditScore',    lambda v: v < 500,         'Poor credit score (< 500)',          -20),
    ('CreditScore',    lambda v: v < 600,         'Below-average credit (< 600)',        -10),
    ('LoanToIncome',   lambda v: v > 1.0,         'Loan > annual income',               -15),
    ('LoanToIncome',   lambda v: v > 0.7,         'High loan-to-income ratio',          -8),
    ('Income',         lambda v: v < 30000,       'Low income (< $30,000)',             -12),
    ('YearsExperience',lambda v: v < 2,           'Less than 2 years experience',       -8),
]

# Positive — score bonuses
POSITIVE_RULES = [
    ('CreditScore',    lambda v: v >= 750,        'Excellent credit score (≥ 750)',     +20),
    ('CreditScore',    lambda v: v >= 700,        'Good credit score (700–749)',         +12),
    ('CreditScore',    lambda v: v >= 650,        'Above-average credit (650–699)',      +6),
    ('Income',         lambda v: v > 80000,       'High income (> $80,000)',            +15),
    ('Income',         lambda v: v > 55000,       'Above-median income (> $55,000)',    +8),
    ('LoanToIncome',   lambda v: v < 0.3,         'Very low loan-to-income ratio',      +15),
    ('LoanToIncome',   lambda v: v < 0.5,         'Low loan-to-income ratio',           +8),
    ('YearsExperience',lambda v: v >= 10,         '10+ years work experience',          +10),
    ('YearsExperience',lambda v: v >= 5,          '5+ years work experience',           +5),
    ('EmploymentType', lambda v: v == 'Salaried', 'Stable salaried employment',         +10),
    ('Education',      lambda v: v in ('PhD','Masters'), 'Advanced education (Masters/PhD)', +5),
]


@dataclass
class RuleDecision:
    hard_reject:   bool = False
    hard_reasons:  List[str] = field(default_factory=list)
    flags:         List[str] = field(default_factory=list)
    positives:     List[str] = field(default_factory=list)
    score:         int = 100
    risk_category: str = 'MEDIUM'
    recommendation:str = 'REVIEW'


def run_rules(applicant: dict) -> RuleDecision:
    dec = RuleDecision()

    # Hard rejects
    for key, cond, reason in HARD_REJECT_RULES:
        val = applicant.get(key)
        if val is not None:
            try:
                if cond(val if isinstance(val, str) else float(val)):
                    dec.hard_reject = True
                    dec.hard_reasons.append(reason)
            except Exception:
                pass

    if dec.hard_reject:
        dec.recommendation = 'REJECT'
        dec.risk_category  = 'CRITICAL'
        dec.score = max(0, dec.score - 60)
        return dec

    # Soft flags
    for key, cond, reason, delta in SOFT_FLAG_RULES:
        val = applicant.get(key)
        if val is not None:
            try:
                if cond(val if isinstance(val, str) else float(val)):
                    dec.flags.append(reason)
                    dec.score += delta
            except Exception:
                pass

    # Positives
    for key, cond, reason, delta in POSITIVE_RULES:
        val = applicant.get(key)
        if val is not None:
            try:
                if cond(val if isinstance(val, str) else float(val)):
                    dec.positives.append(reason)
                    dec.score += delta
            except Exception:
                pass

    dec.score = max(0, min(200, dec.score))

    if dec.score >= 130:
        dec.risk_category = 'LOW';       dec.recommendation = 'APPROVE'
    elif dec.score >= 100:
        dec.risk_category = 'MEDIUM';    dec.recommendation = 'APPROVE'
    elif dec.score >= 75:
        dec.risk_category = 'MEDIUM-HIGH'; dec.recommendation = 'REVIEW'
    else:
        dec.risk_category = 'HIGH';      dec.recommendation = 'REJECT'

    return dec


def combined_decision(applicant: dict, ml_proba: float) -> dict:
    """Blend rule score (60%) with ML probability (40%)."""
    rule_dec = run_rules(applicant)
    ml_risk  = (1 - ml_proba) * 100

    blend = round(rule_dec.score * 0.60 + ml_risk * 0.40, 1)

    if rule_dec.hard_reject:
        final, risk = 'REJECT', 'CRITICAL'
    elif blend >= 115:
        final, risk = 'APPROVE', 'LOW'
    elif blend >= 95:
        final, risk = 'APPROVE', 'MEDIUM'
    elif blend >= 75:
        final, risk = 'REVIEW',  'MEDIUM-HIGH'
    else:
        final, risk = 'REJECT',  'HIGH'

    return {
        'decision':        final,
        'risk_level':      risk,
        'rule_score':      rule_dec.score,
        'ml_default_prob': round(ml_proba * 100, 1),
        'blended_score':   blend,
        'hard_reject':     rule_dec.hard_reject,
        'hard_reasons':    rule_dec.hard_reasons,
        'flags':           rule_dec.flags,
        'positives':       rule_dec.positives,
    }