"""
WEEK 3 — Model Training on REAL dataset
Features: Age, Income, LoanAmount, CreditScore, YearsExperience,
          LoanToIncome, Gender, Education, City, EmploymentType
Target:   LoanApproved (0/1)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, json

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score,
                             precision_recall_curve, average_precision_score)
from sklearn.model_selection import cross_val_score
from sklearn.utils.class_weight import compute_class_weight
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.preprocessor import load_and_split, NUMERICAL_COLS, CATEGORICAL_COLS

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets')
BG, CARD_BG, TEXT = '#0f1117', '#1a1f2e', '#e8eaf6'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD_BG,
    'axes.edgecolor': '#2d3561', 'axes.labelcolor': TEXT,
    'xtick.color': TEXT, 'ytick.color': TEXT, 'text.color': TEXT,
    'grid.color': '#2d3561',
})


def train_models(csv_path: str):
    X_train, X_test, y_train, y_test, _, _ = load_and_split(csv_path)
    features = NUMERICAL_COLS + CATEGORICAL_COLS

    # Class imbalance: ~77% rejected, ~23% approved — use class_weight='balanced'
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=4,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.08,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42, C=0.5
        )
    }

    results = {}
    best_auc, best_name, best_model = 0, None, None

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc  = roc_auc_score(y_test, y_proba)
        acc  = accuracy_score(y_test, y_pred)
        ap   = average_precision_score(y_test, y_proba)
        cv   = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
        results[name] = {'model':model,'auc':auc,'acc':acc,'ap':ap,'cv_auc':cv,
                         'y_pred':y_pred,'y_proba':y_proba}
        print(f"  AUC:{auc:.4f}  Acc:{acc:.4f}  AP:{ap:.4f}  CV-AUC:{cv:.4f}")
        if auc > best_auc:
            best_auc, best_name, best_model = auc, name, model

    print(f"\n🏆 Best: {best_name} (AUC={best_auc:.4f})")
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'loan_model.pkl'))

    meta = {
        'best_model': best_name,
        'auc':        round(best_auc, 4),
        'accuracy':   round(results[best_name]['acc'], 4),
        'avg_precision': round(results[best_name]['ap'], 4),
        'cv_auc':     round(results[best_name]['cv_auc'], 4),
        'features':   features,
        'class_distribution': {'0_rejected': int((y_train==0).sum()),
                               '1_approved': int((y_train==1).sum())}
    }
    with open(os.path.join(MODELS_DIR, 'model_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # ── ROC Curves ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#3498db','#2ecc71','#e74c3c']
    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC={res['auc']:.3f})")
    ax.plot([0,1],[0,1],'--',color='#888',lw=1)
    ax.set_title('ROC Curves — Model Comparison', fontsize=13)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.legend(facecolor=CARD_BG, edgecolor='#2d3561', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR,'model_roc.png'), dpi=120, bbox_inches='tight')
    plt.close()

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, results[best_name]['y_pred'])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Rejected','Approved']); ax.set_yticklabels(['Rejected','Approved'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    fontsize=14, color='white' if cm[i,j]>cm.max()/2 else TEXT)
    ax.set_title(f'Confusion Matrix — {best_name}', fontsize=12)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR,'model_cm.png'), dpi=120, bbox_inches='tight')
    plt.close()

    # ── Feature Importance ────────────────────────────────────────────────────
    if hasattr(best_model, 'feature_importances_'):
        fi = pd.Series(best_model.feature_importances_, index=features).sort_values()
        fig, ax = plt.subplots(figsize=(8, 6))
        colors_fi = ['#e74c3c' if v>fi.quantile(0.75) else
                     '#f39c12' if v>fi.quantile(0.5) else '#3498db'
                     for v in fi.values]
        ax.barh(fi.index, fi.values, color=colors_fi, edgecolor='none')
        ax.set_title('Feature Importance', fontsize=13)
        ax.set_xlabel('Importance Score'); ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(ASSETS_DIR,'model_fi.png'), dpi=120, bbox_inches='tight')
        plt.close()

    # ── Precision-Recall Curve ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for (name, res), color in zip(results.items(), colors):
        prec, rec, _ = precision_recall_curve(y_test, res['y_proba'])
        ax.plot(rec, prec, color=color, lw=2,
                label=f"{name} (AP={res['ap']:.3f})")
    ax.set_title('Precision-Recall Curves', fontsize=13)
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.legend(facecolor=CARD_BG, edgecolor='#2d3561', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR,'model_pr.png'), dpi=120, bbox_inches='tight')
    plt.close()

    print("\n✅ Models saved. Classification report:")
    print(classification_report(y_test, results[best_name]['y_pred'],
                                target_names=['Rejected','Approved']))
    return best_model, meta


if __name__ == '__main__':
    train_models('data/loan_data_raw.csv')