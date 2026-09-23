"""
Health & Summary Routers
"""

from fastapi import APIRouter
from backend.config import settings
from backend.services.state import state
from backend.services.graph_service import get_system_summary
from backend.schemas.health import HealthResponse
from backend.schemas.graph import SystemSummaryResponse

router = APIRouter(prefix="/api", tags=["Health & System"])

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Get API Health Status",
    description="Returns operational status of backend service and loaded state artifacts."
)
def get_health():
    return HealthResponse(
        status="ok",
        service=settings.PROJECT_TITLE,
        version=settings.PROJECT_VERSION,
        model_loaded=(state.model is not None),
        graph_loaded=(state.nodes_df is not None and state.edges_df is not None)
    )

@router.get(
    "/summary",
    response_model=SystemSummaryResponse,
    summary="Get System Summary Overview",
    description="Returns high-level aggregate summary counts derived dynamically from loaded model and graph artifacts."
)
def get_summary():
    return get_system_summary()
