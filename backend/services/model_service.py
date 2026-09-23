"""
Model Service Module
Executes real ML inference using packaged Phase 8A scikit-learn pipeline.
"""

from fastapi import HTTPException
import numpy as np

from backend.services.state import state
from backend.services.drug_service import get_latest_feature_record_for_drug
from backend.schemas.model import RiskPredictionResponse, ModelInfoResponse
from src.model_inference import predict_risk

def get_model_info() -> ModelInfoResponse:
    """
    Return verified model metadata from serialized artifact.
    """
    meta = state.metadata
    return ModelInfoResponse(
        model_name=meta["model_name"],
        model_version=meta["model_version"],
        model_role=meta["model_role"],
        training_feature_years=meta["training_feature_years"],
        target_years=meta["target_years"],
        training_observations=meta["training_observations"],
        training_positive_count=meta["training_positive_count"],
        validation_metrics=meta["validation_metrics"],
        calibrated=meta["calibrated"],
        score_interpretation=meta["score_interpretation"]
    )

def predict_drug_risk_live(node_id: str) -> RiskPredictionResponse:
    """
    Execute REAL ML inference for a canonical drug node.
    1. Find canonical node in graph.
    2. Find latest valid feature observation row.
    3. Execute src.model_inference.predict_risk().
    4. Compare live score against stored node risk score.
    """
    row_matches = state.nodes_df[state.nodes_df['node_id'] == node_id]
    if row_matches.empty:
        raise HTTPException(status_code=404, detail=f"Canonical drug node '{node_id}' not found.")
        
    node_row = row_matches.iloc[0]
    cms_name = node_row['cms_entity_name']
    stored_score = float(node_row['latest_model_risk_score'])
    
    # Retrieve latest feature record from ml_df
    feature_record, feat_year = get_latest_feature_record_for_drug(cms_name)
    
    # Execute REAL ML inference
    live_score = predict_risk(feature_record, model=state.model, schema=state.schema)
    
    # Consistency check against stored node score
    # Note: Stored score in graph CSV is rounded to 4 decimal places (e.g. 0.0447)
    score_diff = abs(live_score - stored_score)
    if score_diff > 0.001:
        consistency = f"WARNING: Material score discrepancy detected ({score_diff:.6f})."
    else:
        consistency = "CONSISTENT: Live inference matches stored graph node risk score within tolerance."
        
    return RiskPredictionResponse(
        node_id=node_id,
        drug_name=cms_name,
        feature_year=feat_year,
        model_name=state.metadata["model_name"],
        base_risk_score=float(live_score),
        stored_graph_risk_score=stored_score,
        consistency_status=consistency,
        score_interpretation="Uncalibrated relative shortage-risk score; not an absolute shortage probability."
    )
