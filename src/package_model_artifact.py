"""
Phase 8A: Validated ML Model Packaging Script

This script deterministically packages the selected Phase 5 Logistic Regression model
(class_weight='balanced', trained on 2018-2021 CMS + FDA data) into a production-ready
sklearn Pipeline artifact along with structured metadata and machine-readable feature schemas.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

RANDOM_SEED = 42

def load_data(filepath="data/processed/ml_feature_candidates.csv"):
    return pd.read_csv(filepath)

def load_feature_dictionary(filepath="data/processed/feature_dictionary.csv"):
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None

def get_feature_lists():
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

def main():
    print("Loading data for model packaging...")
    df = load_data()
    num_cols, cat_cols = get_feature_lists()
    target_col = 'shortage_next_year'
    
    # Filter training split (2018-2021)
    train_mask = df['feature_year_t'].isin([2018, 2019, 2020, 2021])
    val_mask = df['feature_year_t'] == 2022
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    
    X_train, y_train = train_df[num_cols + cat_cols], train_df[target_col]
    X_val, y_val = val_df[num_cols + cat_cols], val_df[target_col]
    
    # Construct exact Phase 5 Pipeline
    preprocessor = build_preprocessor(num_cols, cat_cols)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', random_state=RANDOM_SEED, max_iter=1000))
    ])
    
    print("Fitting production pipeline on Phase 5 training split (2018-2021)...")
    pipeline.fit(X_train, y_train)
    
    # Compute validation metrics
    val_probs = pipeline.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.50).astype(int)
    
    pr_auc = float(average_precision_score(y_val, val_probs))
    roc_auc = float(roc_auc_score(y_val, val_probs))
    prec = float(precision_score(y_val, val_preds))
    rec = float(recall_score(y_val, val_preds))
    f1 = float(f1_score(y_val, val_preds))
    acc = float(accuracy_score(y_val, val_preds))
    
    cm = confusion_matrix(y_val, val_preds)
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    
    print(f"Validation PR-AUC: {pr_auc:.4f} (Ref: 0.2310)")
    print(f"Validation ROC-AUC: {roc_auc:.4f} (Ref: 0.9169)")
    print(f"Validation Precision: {prec:.4f} (Ref: 0.1628)")
    print(f"Validation Recall: {rec:.4f} (Ref: 0.5385)")
    print(f"Validation F1: {f1:.4f} (Ref: 0.2500)")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Serialize Pipeline
    model_path = "models/medcascade_logistic_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Saved serialized pipeline artifact to {model_path}")
    
    # Serialize Metadata
    metadata = {
        "model_name": "MedCascade Rare-Event Logistic Regression Classifier",
        "model_version": "1.0.0",
        "model_role": "experimental rare-event relative shortage-risk classifier",
        "training_feature_years": [2018, 2019, 2020, 2021],
        "target_years": [2019, 2020, 2021, 2022],
        "training_observations": int(len(train_df)),
        "training_positive_count": int(y_train.sum()),
        "validation_feature_year": 2022,
        "validation_observations": int(len(val_df)),
        "validation_positive_count": int(y_val.sum()),
        "feature_names": num_cols + cat_cols,
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "classifier_parameters": {
            "penalty": "l2",
            "C": 1.0,
            "class_weight": "balanced",
            "random_state": RANDOM_SEED,
            "solver": "lbfgs",
            "max_iter": 1000
        },
        "validation_metrics": {
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "accuracy": round(acc, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "decision_threshold": 0.50
        },
        "calibrated": False,
        "score_interpretation": "uncalibrated relative shortage-risk score (bounded in [0, 1])",
        "artifact_creation_info": {
            "framework": "scikit-learn",
            "library_versions": {
                "scikit-learn": joblib.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__
            }
        }
    }
    
    meta_path = "models/medcascade_model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata artifact to {meta_path}")
    
    # Serialize Feature Schema
    feat_dict_df = load_feature_dictionary()
    dict_lookup = {}
    if feat_dict_df is not None:
        for _, row in feat_dict_df.iterrows():
            dict_lookup[row['feature_name']] = {
                'type': row['type'],
                'description': row['description'],
                'source': row['source']
            }
            
    schema_fields = []
    for col in num_cols:
        info = dict_lookup.get(col, {})
        schema_fields.append({
            "name": col,
            "data_type": "float",
            "category": "numerical",
            "required": True,
            "description": info.get('description', 'Numerical feature derived from CMS/FDA records'),
            "source": info.get('source', 'CMS Part D / FDA')
        })
        
    for col in cat_cols:
        info = dict_lookup.get(col, {})
        schema_fields.append({
            "name": col,
            "data_type": "string",
            "category": "categorical",
            "required": True,
            "description": info.get('description', 'Primary FDA therapeutic category classification'),
            "source": info.get('source', 'FDA Shortage')
        })
        
    schema = {
        "schema_name": "MedCascade Inference Input Schema",
        "schema_version": "1.0.0",
        "expected_columns": num_cols + cat_cols,
        "feature_count": len(num_cols + cat_cols),
        "numerical_feature_count": len(num_cols),
        "categorical_feature_count": len(cat_cols),
        "fields": schema_fields,
        "usage_notes": "Input records must match this feature schema exactly before passing to model inference."
    }
    
    schema_path = "models/medcascade_feature_schema.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"Saved feature schema artifact to {schema_path}")

if __name__ == "__main__":
    main()
