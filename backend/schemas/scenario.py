"""
Scenario Stress Simulation Pydantic Schemas
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any

class ScenarioSimulateRequest(BaseModel):
    source_node: str = Field(..., description="Source node ID for stress shock (e.g. NODE_HYDROCHLOROTHIAZIDE)")
    initial_stress: float = Field(1.0, ge=0.0, le=1.0, description="Hypothetical initial stress shock [0, 1]")
    coformulation_weight: float = Field(0.3, ge=0.0, le=1.0, description="Hypothetical co-formulation scenario weight [0, 1]")
    category_weight: float = Field(0.1, ge=0.0, le=1.0, description="Hypothetical category similarity scenario weight [0, 1]")
    decay_factor: float = Field(0.5, ge=0.0, le=1.0, description="Multiplicative attenuation factor per hop [0, 1]")
    max_hops: int = Field(2, ge=1, le=5, description="Maximum propagation distance [1, 5]")
    minimum_pressure_threshold: float = Field(0.01, ge=0.0, le=1.0, description="Minimum pressure cutoff threshold [0, 1]")

class SourceNodeInfo(BaseModel):
    node_id: str
    drug_name: str
    base_risk_score: float

class ScenarioSummary(BaseModel):
    affected_nodes: int
    mean_downstream_pressure: float
    max_downstream_pressure: float

class ScenarioNodeImpact(BaseModel):
    node_id: str
    drug_name: str
    therapeutic_category: str
    base_risk_score: float
    scenario_pressure: float
    scenario_risk_score: float
    hop_distance: Optional[int]

class ScenarioSimulateResponse(BaseModel):
    scenario_configuration: Dict[str, Any]
    source: SourceNodeInfo
    summary: ScenarioSummary
    affected_nodes: List[ScenarioNodeImpact]
    interpretation: str = Field(
        default="Hypothetical inventory stress propagation across an analytical drug relationship network. Results do not represent clinical substitution, patient switching, observed demand transfer, causal shortage transmission, or calibrated shortage probabilities.",
        description="Mandatory scientific boundary disclaimer"
    )
