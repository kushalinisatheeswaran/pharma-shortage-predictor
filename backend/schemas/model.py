"""
Model Information & Real ML Inference Pydantic Schemas
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List

class RiskPredictionResponse(BaseModel):
    node_id: str = Field(..., description="Canonical node identifier")
    drug_name: str = Field(..., description="CMS Entity drug name")
    feature_year: int = Field(..., description="CMS feature observation year evaluated")
    model_name: str = Field(..., description="Name of the packaged ML classifier")
    base_risk_score: float = Field(..., description="Live ML inference relative shortage-risk score [0, 1]")
    stored_graph_risk_score: float = Field(..., description="Reference score stored in graph node metadata")
    consistency_status: str = Field(..., description="Score consistency check status between live ML score and stored graph score")
    score_interpretation: str = Field(
        default="Uncalibrated relative shortage-risk score; not an absolute shortage probability.",
        description="Official scientific interpretation disclaimer"
    )

class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_role: str
    training_feature_years: List[int]
    target_years: List[int]
    training_observations: int
    training_positive_count: int
    validation_metrics: Dict[str, Any]
    calibrated: bool
    score_interpretation: str
