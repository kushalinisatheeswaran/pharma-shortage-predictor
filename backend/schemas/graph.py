"""
Graph & System Summary Pydantic Schemas
"""

from pydantic import BaseModel, Field
from typing import Dict

class GraphSummaryResponse(BaseModel):
    nodes: int = Field(..., description="Total canonical drug nodes count")
    relationship_channels: int = Field(..., description="Total relationship edge channels count")
    unique_node_pairs: int = Field(..., description="Total unique node pair connections")
    connected_nodes: int = Field(..., description="Number of connected drug nodes (degree > 0)")
    isolated_nodes: int = Field(..., description="Number of isolated drug nodes (degree == 0)")
    connected_components: int = Field(..., description="Total connected graph components")
    largest_component_size: int = Field(..., description="Size of largest connected component")
    coformulated_relationships: int = Field(..., description="CO_FORMULATED_WITH relationship channels count")
    therapeutic_category_relationships: int = Field(..., description="SAME_THERAPEUTIC_CATEGORY relationship channels count")

class SystemSummaryResponse(BaseModel):
    total_drugs: int
    mapped_drugs: int
    connected_nodes: int
    isolated_nodes: int
    relationship_channels: int
    unique_node_pairs: int
    relationship_counts: Dict[str, int]
    model_name: str
    model_role: str
    model_calibrated: bool = False
