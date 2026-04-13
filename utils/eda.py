"""
WEEK 2 — EDA using the REAL dataset
Columns: Age, Income, LoanAmount, CreditScore, YearsExperience,
         Gender, Education, City, EmploymentType, LoanApproved
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

ASSETS = os.path.join(os.path.dirname(__file__), '..', 'assets')
os.makedirs(ASSETS, exist_ok=True)

BG, CARD_BG, TEXT = '#0f1117', '#1a1f2e', '#e8eaf6'
PALETTE = {1: '#2ecc71', 0: '#e74c3c'}

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD_BG,
    'axes.edgecolor': '#2d3561', 'axes.labelcolor': TEXT,
    'xtick.color': TEXT, 'ytick.color': TEXT,
    'text.color': TEXT, 'grid.color': '#2d3561',
})


def run_eda(csv_path: str):
    df = pd.read_csv(csv_path)
    df['Income'] = df['Income'].clip(lower=0)
    df['LoanToIncome'] = df['LoanAmount'] / df['Income'].replace(0, np.nan)
    print(f"EDA on {len(df)} rows\n")

    label_map = {1: 'Approved', 0: 'Rejected'}
    df['Status'] = df['LoanApproved'].map(label_map)

    # ── 1. Loan Approval Distribution ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df['LoanApproved'].value_counts().sort_index()
    labels = ['Rejected (0)', 'Approved (1)']
    colors = ['#e74c3c', '#2ecc71']
    bars = ax.bar(labels, counts.values, color=colors, edgecolor='none', width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f'{val:,}\n({val/len(df)*100:.1f}%)', ha='center', fontsize=10)
    ax.set_title('Loan Approval Distribution', fontsize=14, pad=12)
    ax.set_ylabel('Count'); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS, 'eda_loan_status.png'), dpi=120, bbox_inches='tight')
    plt.close()

    # ── 2. Credit Score Distribution by Approval ──────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for status, color in [(1,'#2ecc71'),(0,'#e74c3c')]:
        sub = df[df['LoanApproved']==status]['CreditScore'].dropna()
        ax.hist(sub, bins=40, alpha=0.65, color=color,
                label=label_map[status], edgecolor='none')
    ax.axvline(df[df['LoanApproved']==1]['CreditScore'].mean(),
               color='#2ecc71', linestyle='--', linewidth=1.5, label='Approved mean')
    ax.axvline(df[df['LoanApproved']==0]['CreditScore'].mean(),
               color='#e74c3c', linestyle='--', linewidth=1.5, label='Rejected mean')
    ax.set_title('Credit Score Distribution by Approval Status', fontsize=14, pad=12)
    ax.set_xlabel('Credit Score'); ax.set_ylabel('Frequency')
    ax.legend(facecolor=CARD_BG, edgecolor='#2d3561')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS, 'eda_credit_score.png'), dpi=120, bbox_inches='tight')
    plt.close()

    # ── 3. Income vs Loan Amount scatter ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    sample = df.dropna(subset=['Income','LoanAmount']).sample(min(2000,len(df)), random_state=42)
    for status, color in [(1,'#2ecc71'),(0,'#e74c3c')]:
        sub = sample[sample['LoanApproved']==status]
        ax.scatter(sub['Income'], sub['LoanAmount'], alpha=0.25, s=10,
                   color=color, label=label_map[status])
    ax.set_title('Income vs Loan Amount', fontsize=14, pad=12)
    ax.set_xlabel('Annual Income ($)'); ax.set_ylabel('Loan Amount ($)')
    ax.legend(facecolor=CARD_BG, edgecolor='#2d3561'); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS, 'eda_income_scatter.png'), dpi=120, bbox_inches='tight')
    plt.close()

    # ── 4. Approval Rate by Employment Type ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    emp_rate = df.groupby('EmploymentType')['LoanApproved'].mean() * 100
    colors_e = ['#2ecc71' if v > 20 else '#e74c3c' for v in emp_rate.values]
    bars = ax.bar(emp_rate.index, emp_rate.values, color=colors_e, edgecolor='none')
    for bar, val in zip(bars, emp_rate.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{val:.1f}%', ha='center', fontsize=10)
    ax.set_title('Loan Approval Rate by Employment Type', fontsize=14, pad=12)
    ax.set_ylabel('Approval Rate (%)'); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS, 'eda_employment.png'), dpi=120, bbox_inches='tight')
    plt.close()

    # ── 5. Approval Rate by Education ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    edu_rate = df.groupby('Education')['LoanApproved'].mean() * 100
    edu_rate = edu_rate.sort_values(ascending=True)
    colors_edu = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(edu_rate)))
    ax.barh(edu_rate.index, edu_rate.values, color=colors_edu, edgecolor='none')
    for i, v in enumerate(edu_rate.values):
        ax.text(v+0.3, i, f'{v:.1f}%', va='center', fontsize=9)
    ax.set_title('Approval Rate by Education Level', fontsize=14, pad=12)
    ax.set_xlabel('Approval Rate (%)'); ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS, 'eda_education.png'), dpi=120, bbox_inches='tight')
    plt.close()

    # ── 6. Age Distribution by Approval ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for status, color in [(1,'#2ecc71'),(0,'#e74c3c')]:
        sub = df[df['LoanApproved']==status]['Age']
        ax.hist(sub, bins=30, alpha=0.65, color=color,
                label=label_map[status], edgecolor='none')
    ax.set_title('Age Distribution by Approval Status', fontsize=14, pad=12)
    ax.set_xlabel('Age'); ax.set_ylabel('Count')
    ax.legend(facecolor=CARD_BG, edgecolor='#2d3561'); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS, 'eda_age.png'), dpi=120, bbox_inches='tight')
    plt.close()

    # ── 7. Correlation Heatmap ────────────────────────────────────────────────
    num_cols = ['Age','Income','LoanAmount','CreditScore','YearsExperience','LoanApproved']
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                linewidths=0.5, ax=ax, cbar_kws={'shrink':0.8}, annot_kws={'size':9})
    ax.set_title('Feature Correlation Matrix', fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS, 'eda_correlation.png'), dpi=120, bbox_inches='tight')
    plt.close()

    # ── 8. Approval Rate by City ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    city_rate = df.groupby('City')['LoanApproved'].mean() * 100
    city_colors = ['#3498db','#2ecc71','#e74c3c','#f39c12']
    ax.bar(city_rate.index, city_rate.values, color=city_colors, edgecolor='none')
    for i, v in enumerate(city_rate.values):
        ax.text(i, v+0.4, f'{v:.1f}%', ha='center', fontsize=10)
    ax.set_title('Approval Rate by City', fontsize=14, pad=12)
    ax.set_ylabel('Approval Rate (%)'); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS, 'eda_city.png'), dpi=120, bbox_inches='tight')
    plt.close()

    print("=== Summary Statistics ===")
    print(df[['Age','Income','LoanAmount','CreditScore','YearsExperience']].describe().round(2))
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum()>0]}")
    print("\n✅ All 8 EDA charts saved to assets/")


if __name__ == '__main__':
    run_eda('data/loan_data_raw.csv')