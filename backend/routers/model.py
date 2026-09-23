"""
Model Metadata Router
"""

from fastapi import APIRouter
from backend.schemas.model import ModelInfoResponse
from backend.services.model_service import get_model_info

router = APIRouter(prefix="/api/model", tags=["Model Information"])

@router.get(
    "/info",
    response_model=ModelInfoResponse,
    summary="Get Packaged ML Model Metadata",
    description="Returns verified metadata of the packaged Phase 8A Logistic Regression classifier artifact."
)
def get_model_information():
    return get_model_info()
