"""
Health & System Summary Pydantic Schemas
"""

from pydantic import BaseModel, Field
from typing import Dict

class HealthResponse(BaseModel):
    status: str = Field(..., description="API operational status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    model_loaded: bool = Field(..., description="Flag indicating if ML model artifact is loaded")
    graph_loaded: bool = Field(..., description="Flag indicating if drug graph artifacts are loaded")
