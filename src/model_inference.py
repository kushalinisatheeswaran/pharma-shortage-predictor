"""
Phase 8A: Model Inference Service Module

Provides a lightweight, robust inference layer for loading the packaged MedCascade
Logistic Regression pipeline and scoring single or batch feature vectors.

IMPORTANT SCIENTIFIC & SAFETY DISCLAIMER:
- The output of predict_risk() / predict_batch() is an UNCALIBRATED RELATIVE SHORTAGE-RISK SCORE.
- It is NOT an absolute shortage probability and must NOT be called a calibrated probability.
- Model artifacts are loaded strictly from verified local project paths.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

DEFAULT_MODEL_PATH = "models/medcascade_logistic_pipeline.joblib"
DEFAULT_METADATA_PATH = "models/medcascade_model_metadata.json"
DEFAULT_SCHEMA_PATH = "models/medcascade_feature_schema.json"

_CACHED_PIPELINE = None
_CACHED_METADATA = None
_CACHED_SCHEMA = None

def load_model(model_path=DEFAULT_MODEL_PATH, force_reload=False):
    """
    Load the serialized scikit-learn pipeline from a trusted local path.
    """
    global _CACHED_PIPELINE
    if _CACHED_PIPELINE is not None and not force_reload:
        return _CACHED_PIPELINE
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at '{model_path}'. Ensure Phase 8A packaging script has run.")
        
    _CACHED_PIPELINE = joblib.load(model_path)
    return _CACHED_PIPELINE

def get_model_metadata(metadata_path=DEFAULT_METADATA_PATH, force_reload=False):
    """
    Load structured metadata associated with the model artifact.
    """
    global _CACHED_METADATA
    if _CACHED_METADATA is not None and not force_reload:
        return _CACHED_METADATA
        
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata artifact not found at '{metadata_path}'.")
        
    with open(metadata_path, "r") as f:
        _CACHED_METADATA = json.load(f)
    return _CACHED_METADATA

def get_feature_schema(schema_path=DEFAULT_SCHEMA_PATH, force_reload=False):
    """
    Load machine-readable feature schema for input validation.
    """
    global _CACHED_SCHEMA
    if _CACHED_SCHEMA is not None and not force_reload:
        return _CACHED_SCHEMA
        
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Feature schema artifact not found at '{schema_path}'.")
        
    with open(schema_path, "r") as f:
        _CACHED_SCHEMA = json.load(f)
    return _CACHED_SCHEMA

def validate_feature_input(df, schema=None):
    """
    Verify that all expected input columns are present.
    """
    if schema is None:
        schema = get_feature_schema()
        
    expected_cols = schema["expected_columns"]
    missing_cols = [col for col in expected_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required feature columns for model inference: {missing_cols}")
        
    return df[expected_cols]

def predict_batch(feature_df, model=None, schema=None):
    """
    Score a DataFrame of feature observations.
    
    Returns:
    - numpy array of uncalibrated relative shortage-risk scores in [0, 1].
    """
    if model is None:
        model = load_model()
        
    validated_df = validate_feature_input(feature_df, schema=schema)
    
    # Predict positive class probability (uncalibrated relative risk score)
    scores = model.predict_proba(validated_df)[:, 1]
    scores = np.clip(scores, 0.0, 1.0)
    return scores

def predict_risk(feature_record, model=None, schema=None):
    """
    Score a single feature dictionary or Series.
    
    Returns:
    - float: uncalibrated relative shortage-risk score in [0, 1].
    """
    if isinstance(feature_record, dict):
        df = pd.DataFrame([feature_record])
    elif isinstance(feature_record, pd.Series):
        df = pd.DataFrame([feature_record.to_dict()])
    elif isinstance(feature_record, pd.DataFrame):
        df = feature_record
    else:
        raise TypeError("feature_record must be a dict, pandas Series, or single-row DataFrame.")
        
    scores = predict_batch(df, model=model, schema=schema)
    return float(scores[0])
