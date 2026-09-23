"""
Graph Summary Router
"""

from fastapi import APIRouter
from backend.schemas.graph import GraphSummaryResponse
from backend.services.graph_service import get_graph_summary

router = APIRouter(prefix="/api/graph", tags=["Graph Network"])

@router.get(
    "/summary",
    response_model=GraphSummaryResponse,
    summary="Get Graph B Network Summary",
    description="Returns analytical Graph B topology metrics including total nodes, relationship channels, unique node pairs, and component counts."
)
def get_graph_summary_endpoint():
    return get_graph_summary()
