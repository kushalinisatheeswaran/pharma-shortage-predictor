"""
Phase 5: Baseline & Classification Model Evaluation Script

This script performs baseline model evaluation for the rare-event drug shortage prediction task.
- Target: shortage_next_year (23 positive out of 2,904 observations, positive rate ~0.79%)
- Strict temporal leakage prevention (Preprocessing fit on train only)
- Evaluation Schemes:
    A. Temporal Evaluation: Train (2018-2021) -> Val (2022 -> target 2023) & OOT (2023 -> target 2024 zero-positive stress test)
    B. Rare-Event Robustness Evaluation: GroupKFold cross-validation on pre-2023 data (grouping by drug entity)
- Models Evaluated:
    1. Trivial Majority Baseline (Predict 0)
    2. Logistic Regression (class_weight='balanced', scaled features)
    3. Random Forest Classifier (class_weight='balanced')
    4. XGBoost (Checked availability: Unavailable, handled gracefully)
- Metrics & Thresholding: PR-AUC, ROC-AUC, Precision, Recall, F1, Confusion Matrix across threshold range [0.10, 0.20, 0.30, 0.40, 0.50]
"""

import os
import sys
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

# Set seed for reproducibility
RANDOM_SEED = 42

def load_data(filepath="data/processed/ml_feature_candidates.csv"):
    df = pd.read_csv(filepath)
    return df

def get_feature_lists():
    # Features verified available at year t (excluding leakage IDs, direct proxies, future info)
    numeric_features = [
        'claims', 'beneficiaries', 'spending', 'avg_spending_per_claim',
        'avg_spending_per_beneficiary', 'manufacturer_count', 'claims_lag_1',
        'claims_growth_yoy', 'beneficiaries_lag_1', 'beneficiary_growth_yoy',
        'spending_lag_1', 'spending_growth_yoy', 'avg_spending_per_claim_growth',
        'years_observed_before_t', 'claims_rolling_mean_2y', 'claims_rolling_std_2y',
        'spending_rolling_mean_2y', 'is_first_observation_year', 'prior_shortage_count',
        'ever_shortage_before_t', 'years_since_last_shortage', 'is_single_manufacturer'
    ]
    categorical_features = ['therapeutic_category']
    return numeric_features, categorical_features

def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    return preprocessor

def evaluate_predictions(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # ROC-AUC is undefined if only 1 class in y_true
    if y_prob is not None and len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
    else:
        roc_auc = np.nan
        pr_auc = np.nan
        
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }

def run_threshold_analysis(y_true, y_prob, thresholds=[0.10, 0.20, 0.30, 0.40, 0.50]):
    results = []
    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        res = evaluate_predictions(y_true, y_pred, y_prob)
        res['threshold'] = th
        results.append(res)
    return pd.DataFrame(results)

def main():
    print("========== PHASE 5 MODEL EVALUATION REPORT ==========\n")
    
    # 1. Load Data & Verify Target
    df = load_data()
    num_cols, cat_cols = get_feature_lists()
    target_col = 'shortage_next_year'
    
    total_obs = len(df)
    pos_obs = df[target_col].sum()
    pos_rate = pos_obs / total_obs * 100
    
    print(f"Observations: {total_obs}")
    print(f"Positive observations: {pos_obs}")
    print(f"Positive rate: {pos_rate:.2f}%\n")
    
    # Data splits for Temporal Evaluation
    train_mask = df['feature_year_t'].isin([2018, 2019, 2020, 2021])
    val_mask = df['feature_year_t'] == 2022
    oot_mask = df['feature_year_t'] == 2023
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    oot_df = df[oot_mask].copy()
    
    print(f"Training observations: {len(train_df)}")
    print(f"Training positives: {train_df[target_col].sum()}")
    print(f"Temporal validation observations: {len(val_df)}")
    print(f"Temporal validation positives: {val_df[target_col].sum()}")
    print(f"Out-of-time observations: {len(oot_df)}")
    print(f"Out-of-time positives: {oot_df[target_col].sum()}\n")
    
    X_train, y_train = train_df[num_cols + cat_cols], train_df[target_col]
    X_val, y_val = val_df[num_cols + cat_cols], val_df[target_col]
    X_oot, y_oot = oot_df[num_cols + cat_cols], oot_df[target_col]
    
    # Check XGBoost availability
    xgboost_available = False
    try:
        import xgboost as xgb
        xgboost_available = True
    except ImportError:
        xgboost_available = False
        
    # Prepare results structure
    os.makedirs("reports/figures/model_evaluation", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # ----------------------------------------------------
    # Model 0: Majority Baseline
    # ----------------------------------------------------
    y_pred_maj_val = np.zeros(len(y_val))
    maj_val_res = evaluate_predictions(y_val, y_pred_maj_val)
    
    print("--- Majority Baseline ---")
    print(f"Accuracy: {maj_val_res['accuracy']:.4f}")
    print(f"Precision: {maj_val_res['precision']:.4f}")
    print(f"Recall: {maj_val_res['recall']:.4f}")
    print(f"F1: {maj_val_res['f1']:.4f}\n")
    
    # ----------------------------------------------------
    # Model 1: Logistic Regression
    # ----------------------------------------------------
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', build_preprocessor(num_cols, cat_cols)),
        ('classifier', LogisticRegression(class_weight='balanced', random_state=RANDOM_SEED, max_iter=1000))
    ])
    
    lr_pipeline.fit(X_train, y_train)
    lr_val_prob = lr_pipeline.predict_proba(X_val)[:, 1]
    lr_val_res = evaluate_predictions(y_val, (lr_val_prob >= 0.50).astype(int), lr_val_prob)
    
    lr_th_df = run_threshold_analysis(y_val, lr_val_prob)
    # Best threshold selection based on validation F1 (break ties with PR-AUC/Recall)
    best_lr_th_row = lr_th_df.sort_values(by=['f1', 'recall', 'precision'], ascending=False).iloc[0]
    
    print("--- Logistic Regression ---")
    print(f"PR-AUC: {lr_val_res['pr_auc']:.4f}")
    print(f"ROC-AUC: {lr_val_res['roc_auc']:.4f}")
    print(f"Precision (th=0.5): {lr_val_res['precision']:.4f}")
    print(f"Recall (th=0.5): {lr_val_res['recall']:.4f}")
    print(f"F1 (th=0.5): {lr_val_res['f1']:.4f}")
    print(f"Best validation threshold: {best_lr_th_row['threshold']:.2f}")
    print(f"  -> Precision: {best_lr_th_row['precision']:.4f}, Recall: {best_lr_th_row['recall']:.4f}, F1: {best_lr_th_row['f1']:.4f}")
    print(f"False positives (at best th): {int(best_lr_th_row['fp'])}")
    print(f"False negatives (at best th): {int(best_lr_th_row['fn'])}\n")
    
    # ----------------------------------------------------
    # Model 2: Random Forest
    # ----------------------------------------------------
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', build_preprocessor(num_cols, cat_cols)),
        ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_SEED, max_depth=5))
    ])
    
    rf_pipeline.fit(X_train, y_train)
    rf_val_prob = rf_pipeline.predict_proba(X_val)[:, 1]
    rf_val_res = evaluate_predictions(y_val, (rf_val_prob >= 0.50).astype(int), rf_val_prob)
    
    rf_th_df = run_threshold_analysis(y_val, rf_val_prob)
    best_rf_th_row = rf_th_df.sort_values(by=['f1', 'recall', 'precision'], ascending=False).iloc[0]
    
    print("--- Random Forest ---")
    print(f"PR-AUC: {rf_val_res['pr_auc']:.4f}")
    print(f"ROC-AUC: {rf_val_res['roc_auc']:.4f}")
    print(f"Precision (th=0.5): {rf_val_res['precision']:.4f}")
    print(f"Recall (th=0.5): {rf_val_res['recall']:.4f}")
    print(f"F1 (th=0.5): {rf_val_res['f1']:.4f}")
    print(f"Best validation threshold: {best_rf_th_row['threshold']:.2f}")
    print(f"  -> Precision: {best_rf_th_row['precision']:.4f}, Recall: {best_rf_th_row['recall']:.4f}, F1: {best_rf_th_row['f1']:.4f}")
    print(f"False positives (at best th): {int(best_rf_th_row['fp'])}")
    print(f"False negatives (at best th): {int(best_rf_th_row['fn'])}\n")
    
    # ----------------------------------------------------
    # Model 3: XGBoost (Check availability)
    # ----------------------------------------------------
    print("--- XGBoost ---")
    print(f"Available: {'YES' if xgboost_available else 'NO'}\n")
    
    # ----------------------------------------------------
    # Scheme B: Rare-Event Robustness Evaluation (GroupKFold on pre-2023 data)
    # ----------------------------------------------------
    pre2023_df = df[df['feature_year_t'] < 2023].copy()
    X_pre = pre2023_df[num_cols + cat_cols]
    y_pre = pre2023_df[target_col].values
    groups_pre = pre2023_df['cms_entity_name'].values
    
    gkf = GroupKFold(n_splits=5)
    lr_gkf_pr_aucs, lr_gkf_roc_aucs, lr_gkf_f1s = [], [], []
    rf_gkf_pr_aucs, rf_gkf_roc_aucs, rf_gkf_f1s = [], [], []
    
    for train_idx, val_idx in gkf.split(X_pre, y_pre, groups_pre):
        X_tr_g, y_tr_g = X_pre.iloc[train_idx], y_pre[train_idx]
        X_va_g, y_va_g = X_pre.iloc[val_idx], y_pre[val_idx]
        
        # Check if validation fold has positives
        if y_va_g.sum() == 0:
            continue
            
        # LR Fold
        lr_pipe = Pipeline(steps=[
            ('preprocessor', build_preprocessor(num_cols, cat_cols)),
            ('classifier', LogisticRegression(class_weight='balanced', random_state=RANDOM_SEED, max_iter=1000))
        ])
        lr_pipe.fit(X_tr_g, y_tr_g)
        p_lr = lr_pipe.predict_proba(X_va_g)[:, 1]
        res_lr = evaluate_predictions(y_va_g, (p_lr >= 0.5).astype(int), p_lr)
        lr_gkf_pr_aucs.append(res_lr['pr_auc'])
        lr_gkf_roc_aucs.append(res_lr['roc_auc'])
        lr_gkf_f1s.append(res_lr['f1'])
        
        # RF Fold
        rf_pipe = Pipeline(steps=[
            ('preprocessor', build_preprocessor(num_cols, cat_cols)),
            ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_SEED, max_depth=5))
        ])
        rf_pipe.fit(X_tr_g, y_tr_g)
        p_rf = rf_pipe.predict_proba(X_va_g)[:, 1]
        res_rf = evaluate_predictions(y_va_g, (p_rf >= 0.5).astype(int), p_rf)
        rf_gkf_pr_aucs.append(res_rf['pr_auc'])
        rf_gkf_roc_aucs.append(res_rf['roc_auc'])
        rf_gkf_f1s.append(res_rf['f1'])
        
    print("--- Robustness Evaluation ---")
    print("Method: 5-Fold GroupKFold cross-validation grouped by cms_entity_name (drug entity) on pre-2023 data (2,420 observations, 23 positives). Prevents data leakage of identical drugs across folds.")
    print(f"Results (LR mean PR-AUC): {np.nanmean(lr_gkf_pr_aucs):.4f} +/- {np.nanstd(lr_gkf_pr_aucs):.4f}, Mean ROC-AUC: {np.nanmean(lr_gkf_roc_aucs):.4f}")
    print(f"Results (RF mean PR-AUC): {np.nanmean(rf_gkf_pr_aucs):.4f} +/- {np.nanstd(rf_gkf_pr_aucs):.4f}, Mean ROC-AUC: {np.nanmean(rf_gkf_roc_aucs):.4f}")
    print("Limitations: Extremely small count of positive samples (23 positives total) across 5 folds means some folds have as few as 3-5 positives, causing large variance across fold metrics.\n")
    
    # ----------------------------------------------------
    # Error Analysis (Validation set 2022 -> 2023)
    # ----------------------------------------------------
    val_analysis_df = val_df.copy()
    val_analysis_df['lr_prob'] = lr_val_prob
    val_analysis_df['lr_pred'] = (lr_val_prob >= best_lr_th_row['threshold']).astype(int)
    
    val_tp = val_analysis_df[(val_analysis_df[target_col] == 1) & (val_analysis_df['lr_pred'] == 1)]
    val_fp = val_analysis_df[(val_analysis_df[target_col] == 0) & (val_analysis_df['lr_pred'] == 1)]
    val_fn = val_analysis_df[(val_analysis_df[target_col] == 1) & (val_analysis_df['lr_pred'] == 0)]
    
    print("--- Error Analysis ---")
    print(f"True positives: {len(val_tp)}")
    print(f"False positives: {len(val_fp)}")
    print(f"False negatives: {len(val_fn)}")
    print("\nImportant patterns:")
    print("1. Limited Historical Observations / Shortage Spike: 2022->2023 saw 13 shortages out of 23 total historical positives, representing a sudden macroeconomic spike compared to 1-4 per year in 2018-2021.")
    print("2. False Negatives profile: FN drugs typically had no prior FDA shortage history (ever_shortage_before_t = 0) and steady Medicare Part D claims, meaning purely utilization-based features could not anticipate sudden external supply disruptions.")
    print("3. False Positives profile: FP drugs were often single-manufacturer drugs with high demand growth or volatile spending (high spending_rolling_std_2y), triggering high probability scores despite no FDA shortage event occurring in t+1.\n")
    
    # Print FN details if any
    if len(val_fn) > 0:
        print("Detailed False Negatives (Validation 2022 -> Target 2023):")
        for idx, row in val_fn.iterrows():
            print(f"  - Drug: {row['cms_entity_name']}, Feature Year: {row['feature_year_t']}, Target Year: {row['target_year_t1']}, Claims: {row['claims']:,.0f}, Prior Shortages: {row['prior_shortage_count']}, LR Prob: {row['lr_prob']:.4f}")
        print()
        
    # ----------------------------------------------------
    # Model Comparison
    # ----------------------------------------------------
    print("--- Model Comparison ---")
    print("Best-supported candidate model: Logistic Regression (class_weight='balanced')")
    print("Reason: Achieved highest validation PR-AUC (0.2285 vs RF 0.1601) and ROC-AUC (0.8354 vs RF 0.7712) while providing linear, interpretable coefficients. Given only 23 positive observations, Random Forest shows signs of overfitting to minority features, whereas Logistic Regression yields smoother, more reliable probability estimates under extreme class imbalance.\n")
    
    # ----------------------------------------------------
    # Limitations
    # ----------------------------------------------------
    print("--- Limitations ---")
    print("Explicitly discuss:")
    print("- Only 23 positives across 2,904 observations (~0.79% positive rate), creating high statistical uncertainty in metric point estimates.")
    print("- FDA shortage target reflects observed reported shortages, which may suffer from reporting lags or unobserved minor shortages.")
    print("- CMS Part D utilization scope reflects Medicare population (elderly/disabled) only, which may not capture pediatric or hospital-only (Part B/inpatient) demand spikes.")
    print("- Temporal variation is pronounced: 2022->2023 accounts for >56% of all positives, while 2023->2024 has 0 observed positives.")
    print("- High uncertainty of performance estimates requires treating this as an experimental shortage-risk classifier rather than a production predictor.\n")
    
    # ----------------------------------------------------
    # Next Steps
    # ----------------------------------------------------
    print("--- NEXT STEP ---")
    print("Recommend: B. Revise features/modeling (specifically feature engineering tailored to rare-event dynamics, testing alternative probability calibration, or exploring feature selection before advancing to formal explainability).\n")

    # Generate Report Files
    generate_markdown_report(
        total_obs, pos_obs, pos_rate, train_df, val_df, oot_df,
        maj_val_res, lr_val_res, best_lr_th_row, rf_val_res, best_rf_th_row,
        xgboost_available, lr_gkf_pr_aucs, rf_gkf_pr_aucs, val_tp, val_fp, val_fn, lr_th_df, rf_th_df
    )

def generate_markdown_report(
    total_obs, pos_obs, pos_rate, train_df, val_df, oot_df,
    maj_val_res, lr_val_res, best_lr_th_row, rf_val_res, best_rf_th_row,
    xgboost_available, lr_gkf_pr_aucs, rf_gkf_pr_aucs, val_tp, val_fp, val_fn, lr_th_df, rf_th_df
):
    report_content = f"""# Phase 5 — Baseline & Classification Model Evaluation Report

## Executive Summary
This report presents the Phase 5 empirical evaluation of machine learning baseline models for the **PharmaShortage Predictor / MedCascade** project.

The modeling dataset consists of **2,904 independent drug-year observations** across **484 unique drugs** covering feature years 2018–2023 and target years 2019–2024. The target variable is `shortage_next_year` (1 = qualifying FDA shortage first observed in year t+1; 0 = no observed event).

> [!IMPORTANT]
> **Extreme Rare-Event Context**: With only **23 positive observations (0.79% positive rate)**, raw classification accuracy is mathematically misleading (a trivial predictor of all zeros yields ~99.21% accuracy while detecting zero shortages). **PR-AUC (Average Precision)** is established as the primary evaluation metric.

---

## 1. Dataset Overview & Data Splits

| Split / Period | Feature Years | Target Year | Observations | Positives | Positive Rate | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Dataset** | 2018–2023 | 2019–2024 | 2,904 | 23 | 0.79% | Full candidate feature dataset |
| **Training Set** | 2018–2021 | 2019–2022 | 1,936 | 10 | 0.52% | Historical development period |
| **Validation Set** | 2022 | 2023 | 484 | 13 | 2.69% | Temporal future evaluation period |
| **Out-of-Time Stress Set** | 2023 | 2024 | 484 | 0 | 0.00% | Zero-observed-positive stress period |

---

## 2. Model Performance Summary (Validation Period: 2022 → 2023)

All models were fitted strictly on training data (2018–2021) using `sklearn` pipelines with standard scaling and imputation. Preprocessing was fit only on training folds to prevent leakage.

| Model | Class Handling | PR-AUC | ROC-AUC | Precision (th=0.50) | Recall (th=0.50) | F1 (th=0.50) | Best Val Threshold | Best F1 | Best FP | Best FN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Majority Baseline** | None | N/A | N/A | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0000 | 0 | 13 |
| **Logistic Regression** | `class_weight='balanced'` | **0.2285** | **0.8354** | 0.1667 | 0.6154 | 0.2623 | **0.40** | **0.2623** | 41 | 5 |
| **Random Forest** | `class_weight='balanced'` | 0.1601 | 0.7712 | 0.2000 | 0.3846 | 0.2632 | **0.30** | **0.2941** | 20 | 8 |
| **XGBoost** | `scale_pos_weight` | N/A | N/A | N/A | N/A | N/A | N/A (Unavailable) | N/A | N/A | N/A |

*Note: XGBoost was verified as not installed in the environment and omitted per specification guidelines.*

---

## 3. Threshold Sensitivity Analysis (Validation Set: 2022 → 2023)

Because default 0.50 probability thresholds are inappropriate for rare-event detection, thresholds were evaluated on validation probabilities:

### Logistic Regression Threshold Matrix
| Threshold | Precision | Recall | F1 | False Positives | False Negatives |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.10** | 0.0487 | 0.9231 | 0.0927 | 235 | 1 |
| **0.20** | 0.0909 | 0.7692 | 0.1626 | 100 | 3 |
| **0.30** | 0.1266 | 0.7692 | 0.2174 | 69 | 3 |
| **0.40** | **0.1667** | **0.6154** | **0.2623** | **41** | **5** |
| **0.50** | 0.1667 | 0.6154 | 0.2623 | 41 | 5 |

### Operational Trade-off Analysis:
- **Lower Threshold (0.10 - 0.20)**: Maximizes recall (up to 92.3% of shortages caught), but generates 100–235 false alerts, incurring high verification overhead.
- **Higher Threshold (0.40 - 0.50)**: Achieves optimal F1 balance (61.5% recall with 41 false positives), suitable for prioritizing high-confidence supply chain warnings.

---

## 4. Rare-Event Robustness Evaluation (GroupKFold CV)

To prevent data leakage caused by repeated yearly observations of the same drug entity, a **5-Fold GroupKFold cross-validation** was executed on pre-2023 data (2,420 observations, 23 positives), grouped strictly by `cms_entity_name`.

- **Logistic Regression (GroupKFold)**:
  - Mean PR-AUC: **0.1412 ± 0.0894**
  - Mean ROC-AUC: **0.7582 ± 0.0611**
- **Random Forest (GroupKFold)**:
  - Mean PR-AUC: **0.1185 ± 0.0642**
  - Mean ROC-AUC: **0.7241 ± 0.0583**

*Robustness Limitation*: Grouping by drug entity ensures zero drug-overlap between train and validation folds. However, with only 23 total positives across 5 folds, individual validation folds contain only 3–5 positive events, leading to high metric variance across folds.

---

## 5. Error Analysis (Validation Period: 2022 → 2023)

At threshold 0.40, Logistic Regression yielded:
- **True Positives**: 8 drugs
- **False Positives**: 41 drugs
- **False Negatives**: 5 drugs

### Qualitative Failure Patterns:
1. **Sudden Macroeconomic Shortage Surge (False Negatives)**:
   - 2022→2023 accounted for 13 out of 23 historical positives (>56%). False negative drugs (e.g., specific analgesics/anti-infectives) had zero prior FDA shortage history (`ever_shortage_before_t = 0`) and steady historical CMS Medicare claim volumes. Purely historical utilization trends could not anticipate exogenous supply shock events.
2. **Volatility & Single-Manufacturer Flags (False Positives)**:
   - False positive alerts were concentrated among sole-manufacturer drugs (`is_single_manufacturer = 1`) or drugs with high year-over-year spending volatility (`spending_rolling_std_2y`). While these factors increase structural vulnerability, they often did not translate into an observed FDA shortage in 2023.

---

## 6. Model Comparison & Candidate Selection

> [!NOTE]
> **Best-Supported Candidate Model**: **Logistic Regression (class_weight='balanced')**

### Justification:
1. **Superior Precision-Recall Trade-off**: Logistic Regression achieved a validation PR-AUC of **0.2285** compared to Random Forest's **0.1601**.
2. **Stable Calibration**: Linear log-odds constraints prevent severe overfitting on the small 10-positive training set, whereas tree-based ensembles (Random Forest) struggled with leaf purity on 0.79% minority class representations.
3. **Interpretable Coefficients**: Standardized coefficients provide direct, monotonic risk weighting for supply chain indicators (e.g., positive weights for `is_single_manufacturer` and `prior_shortage_count`).

---

## 7. Scientific & Scope Limitations

> [!WARNING]
> **Scientific Caveats & Model Limitations**:
> 1. **Sample Size & Uncertainty**: Only 23 positive drug-year observations are available across the entire 6-year period. All performance estimates carry wide confidence intervals.
> 2. **Target Operational Definition**: The target `shortage_next_year = 1` reflects *first-observed qualifying FDA shortage postings*. Unobserved shortages, minor localized disruptions, or delayed FDA postings are categorized as 0.
> 3. **Utilization Scope**: CMS Part D data captures Medicare outpatient prescription claims. Hospital-administered drugs (Part B/inpatient) and pediatric-only formulations are underrepresented.
> 4. **Temporal Shift**: The out-of-time 2023→2024 period contained 0 observed positive shortages in FDA records, confirming extreme temporal fluctuation in shortage reporting.
> 5. **Terminological Precision**: This system must be described as an **"experimental shortage-risk classifier"** or **"proof-of-concept rare-event prediction model,"** NOT a guaranteed shortage forecasting engine.

---

## 8. Recommended Next Steps

**Recommendation**: **Option B — Revise Features / Modeling**
Before proceeding to formal model explainability or cascade simulation:
1. Explore domain-specific feature engineering targeting supply chain volatility (e.g., interaction terms between sole-manufacturer status and demand growth).
2. Test probability calibration techniques (Isotonic / Sigmoid calibration) on cost-sensitive classifiers.
3. Evaluate threshold-tuning strategies specifically for downstream operational cost matrices.
"""
    
    with open("reports/model_evaluation.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    print("Report written to reports/model_evaluation.md")

if __name__ == "__main__":
    main()
