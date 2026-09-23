"""
Phase 8A: Model Inference Unit Tests

Comprehensive unit test suite for verifying model serialization, loading, prediction reproducibility,
feature validation, and pipeline compatibility.
"""

import os
import sys
sys.path.insert(0, ".")
try:
    import pytest
except ImportError:
    pytest = None
import numpy as np
import pandas as pd

from src.model_inference import (
    load_model, get_model_metadata, get_feature_schema,
    predict_risk, predict_batch, validate_feature_input
)

def test_model_artifact_loading():
    """Verify that model, metadata, and feature schema load successfully."""
    model = load_model(force_reload=True)
    assert model is not None
    assert hasattr(model, "predict_proba")
    
    metadata = get_model_metadata(force_reload=True)
    assert metadata["model_version"] == "1.0.0"
    assert metadata["calibrated"] is False
    
    schema = get_feature_schema(force_reload=True)
    assert len(schema["expected_columns"]) == 23

def test_single_row_prediction():
    """Verify single feature record scoring and score bounds."""
    df = pd.read_csv("data/processed/ml_feature_candidates.csv")
    sample_record = df.iloc[0].to_dict()
    
    score = predict_risk(sample_record)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

def test_batch_prediction():
    """Verify batch scoring on full dataset split."""
    df = pd.read_csv("data/processed/ml_feature_candidates.csv")
    val_df = df[df['feature_year_t'] == 2022].copy()
    
    scores = predict_batch(val_df)
    assert len(scores) == len(val_df)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()

def test_missing_feature_validation():
    """Verify controlled error when required feature is missing."""
    df = pd.read_csv("data/processed/ml_feature_candidates.csv")
    sample_record = df.iloc[0].to_dict()
    
    # Remove a required column
    del sample_record['claims']
    
    error_caught = False
    try:
        predict_risk(sample_record)
    except ValueError as e:
        error_caught = True
        assert "Missing required feature columns" in str(e)
    assert error_caught is True

def test_reproducibility_against_phase_5_reference():
    """Verify exact match of temporal validation metrics and zero class mismatches."""
    df = pd.read_csv("data/processed/ml_feature_candidates.csv")
    val_df = df[df['feature_year_t'] == 2022].copy()
    y_val = val_df['shortage_next_year'].values
    
    scores = predict_batch(val_df)
    preds = (scores >= 0.50).astype(int)
    
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
    
    pr_auc = average_precision_score(y_val, scores)
    roc_auc = roc_auc_score(y_val, scores)
    prec = precision_score(y_val, preds)
    rec = recall_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    cm = confusion_matrix(y_val, preds)
    tn, fp, fn, tp = cm.ravel()
    
    # Check exact reference metrics
    assert round(pr_auc, 4) == 0.2310
    assert round(roc_auc, 4) == 0.9169
    assert round(prec, 4) == 0.1628
    assert round(rec, 4) == 0.5385
    assert round(f1, 4) == 0.2500
    assert (tp, fp, fn) == (7, 36, 6)

def main():
    print("Executing unit tests for model_inference.py...")
    test_model_artifact_loading()
    print("[OK] Model artifact loading passed")
    test_single_row_prediction()
    print("[OK] Single row prediction passed")
    test_batch_prediction()
    print("[OK] Batch prediction passed")
    test_missing_feature_validation()
    print("[OK] Missing feature validation passed")
    test_reproducibility_against_phase_5_reference()
    print("[OK] Reproducibility against Phase 5 reference passed")
    print("\nAll unit tests passed successfully!")

if __name__ == "__main__":
    main()
