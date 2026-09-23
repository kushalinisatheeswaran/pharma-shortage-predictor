"""
Drugs & Real ML Risk Inference Routers
"""

from typing import Optional
from fastapi import APIRouter, Query
from backend.schemas.drug import DrugListResponse, DrugDetail, DrugRelationshipsResponse
from backend.schemas.model import RiskPredictionResponse
from backend.services.drug_service import search_drugs, get_drug_detail, get_drug_relationships
from backend.services.model_service import predict_drug_risk_live

router = APIRouter(prefix="/api", tags=["Drugs & ML Risk"])

@router.get(
    "/drugs",
    response_model=DrugListResponse,
    summary="Search & Browse Canonical Drugs",
    description="Query canonical drug nodes with case-insensitive search, category filtering, and pagination."
)
def get_drugs(
    search: Optional[str] = Query(None, description="Case-insensitive search query"),
    category: Optional[str] = Query(None, description="Filter by FDA therapeutic category"),
    connected_only: bool = Query(False, description="Filter to drugs with graph relationships (degree > 0)"),
    limit: int = Query(50, ge=1, le=500, description="Pagination limit"),
    offset: int = Query(0, ge=0, description="Pagination offset")
):
    return search_drugs(
        search=search,
        category=category,
        connected_only=connected_only,
        limit=limit,
        offset=offset
    )

@router.get(
    "/drugs/{node_id}",
    response_model=DrugDetail,
    summary="Get Drug Details",
    description="Retrieve canonical metadata, manufacturer count, shortage history, and relationship counts for a specific drug node."
)
def get_drug_by_id(node_id: str):
    return get_drug_detail(node_id)

@router.get(
    "/risk/{node_id}",
    response_model=RiskPredictionResponse,
    summary="Execute Real ML Shortage Risk Inference",
    description="Executes the packaged Phase 8A scikit-learn pipeline on the latest valid CMS feature vector for the target drug."
)
def get_drug_risk_inference(node_id: str):
    return predict_drug_risk_live(node_id)

@router.get(
    "/drugs/{node_id}/relationships",
    response_model=DrugRelationshipsResponse,
    summary="Get Drug Relationship Channels",
    description="Retrieve Graph B relationship channels connected to a drug node. Supports filtering by relationship type."
)
def get_drug_relationships_by_id(
    node_id: str,
    relationship_type: Optional[str] = Query(None, description="Filter by CO_FORMULATED_WITH or SAME_THERAPEUTIC_CATEGORY")
):
    return get_drug_relationships(node_id=node_id, relationship_type=relationship_type)
