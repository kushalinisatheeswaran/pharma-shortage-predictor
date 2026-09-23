"""
Drug Pydantic Schemas
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class DrugSummary(BaseModel):
    node_id: str = Field(..., description="Canonical node identifier (e.g. NODE_AMOXICILLIN)")
    drug_name: str = Field(..., description="CMS Entity drug name")
    therapeutic_category: str = Field(..., description="FDA therapeutic category classification")
    latest_feature_year: int = Field(..., description="Latest available CMS feature observation year")
    base_risk_score: float = Field(..., description="Uncalibrated relative model risk score [0, 1]")
    relationship_count: int = Field(..., description="Total connected relationship channels")
    is_connected: bool = Field(..., description="Flag indicating if drug has any graph relationships")

class DrugListResponse(BaseModel):
    total: int = Field(..., description="Total matching drugs count")
    limit: int = Field(..., description="Pagination limit")
    offset: int = Field(..., description="Pagination offset")
    drugs: List[DrugSummary] = Field(..., description="List of drug summaries")

class DrugDetail(DrugSummary):
    canonical_rxcui: str = Field(..., description="Canonical RxCUI identifier or UNMAPPED")
    mapping_status: str = Field(..., description="RxNorm mapping status")
    manufacturer_count: int = Field(..., description="Number of active manufacturers")
    single_manufacturer_flag: int = Field(..., description="Sole manufacturer reliance flag")
    historical_shortage_count: int = Field(..., description="Historical FDA shortage event count")
    relationship_counts_by_type: Dict[str, int] = Field(..., description="Counts breakdown by relationship type")

class DrugRelationshipItem(BaseModel):
    related_node_id: str = Field(..., description="Connected target node identifier")
    drug_name: str = Field(..., description="Connected drug name")
    relationship_type: str = Field(..., description="Relationship type (CO_FORMULATED_WITH or SAME_THERAPEUTIC_CATEGORY)")
    base_risk_score: float = Field(..., description="Target drug base relative risk score")

class DrugRelationshipsResponse(BaseModel):
    node_id: str = Field(..., description="Source node identifier")
    drug_name: str = Field(..., description="Source drug name")
    total_relationships: int = Field(..., description="Total relationship channels returned")
    relationships: List[DrugRelationshipItem] = Field(..., description="List of relationship channels")
