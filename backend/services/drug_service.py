"""
Drug Service Module
"""

from typing import Optional, List, Dict, Tuple
from fastapi import HTTPException
import pandas as pd

from backend.services.state import state
from backend.schemas.drug import (
    DrugSummary, DrugDetail, DrugListResponse,
    DrugRelationshipItem, DrugRelationshipsResponse
)

BASELINE_FEATURE_YEAR = 2022

def get_latest_feature_record_for_drug(cms_entity_name: str) -> Tuple[pd.Series, int]:
    """
    Retrieve feature observation for a drug entity for the MedCascade production baseline feature year (2022).
    """
    entity_df = state.ml_df[
        (state.ml_df['cms_entity_name'] == cms_entity_name) & 
        (state.ml_df['feature_year_t'] == BASELINE_FEATURE_YEAR)
    ]
    if entity_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No ML feature observation record found for drug entity '{cms_entity_name}' for production baseline feature year {BASELINE_FEATURE_YEAR}."
        )
    feature_row = entity_df.iloc[0]
    return feature_row, BASELINE_FEATURE_YEAR

def search_drugs(
    search: Optional[str] = None,
    category: Optional[str] = None,
    connected_only: bool = False,
    limit: int = 50,
    offset: int = 0
) -> DrugListResponse:
    """
    Search and filter canonical drugs from loaded graph nodes.
    """
    df = state.nodes_df.copy()
    
    if search:
        search_lower = search.strip().lower()
        df = df[df['cms_entity_name'].str.lower().str.contains(search_lower)]
        
    if category:
        cat_lower = category.strip().lower()
        df = df[df['therapeutic_category'].str.lower().str.contains(cat_lower)]
        
    df['degree'] = df['node_id'].map(state.simulator.degrees)
    
    if connected_only:
        df = df[df['degree'] > 0]
        
    # Sort deterministically by node_id
    df = df.sort_values('node_id', ascending=True)
    
    total = len(df)
    paginated_df = df.iloc[offset:offset+limit]
    
    drug_summaries = []
    for _, row in paginated_df.iterrows():
        nid = row['node_id']
        entity_name = row['cms_entity_name']
        deg = state.simulator.degrees[nid]
        
        # Get latest feature year
        latest_rows = state.latest_feature_obs[state.latest_feature_obs['cms_entity_name'] == entity_name]
        feat_year = int(latest_rows.iloc[0]['feature_year_t']) if not latest_rows.empty else 2022
        
        drug_summaries.append(DrugSummary(
            node_id=nid,
            drug_name=entity_name,
            therapeutic_category=row['therapeutic_category'],
            latest_feature_year=feat_year,
            base_risk_score=float(row['latest_model_risk_score']),
            relationship_count=deg,
            is_connected=(deg > 0)
        ))
        
    return DrugListResponse(
        total=total,
        limit=limit,
        offset=offset,
        drugs=drug_summaries
    )

def get_drug_detail(node_id: str) -> DrugDetail:
    """
    Retrieve comprehensive details for a specific canonical drug node.
    """
    row_matches = state.nodes_df[state.nodes_df['node_id'] == node_id]
    if row_matches.empty:
        raise HTTPException(status_code=404, detail=f"Canonical drug node '{node_id}' not found.")
        
    row = row_matches.iloc[0]
    entity_name = row['cms_entity_name']
    deg = state.simulator.degrees[node_id]
    
    # Relationship breakdown by type
    adj_edges = state.simulator.adjacency[node_id]
    rel_counts = {}
    for edge in adj_edges:
        rel_type = edge['type']
        rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1
        
    # Latest feature year
    latest_rows = state.latest_feature_obs[state.latest_feature_obs['cms_entity_name'] == entity_name]
    feat_year = int(latest_rows.iloc[0]['feature_year_t']) if not latest_rows.empty else 2022
    
    return DrugDetail(
        node_id=node_id,
        drug_name=entity_name,
        therapeutic_category=row['therapeutic_category'],
        latest_feature_year=feat_year,
        base_risk_score=float(row['latest_model_risk_score']),
        relationship_count=deg,
        is_connected=(deg > 0),
        canonical_rxcui=str(row['canonical_rxcui']),
        mapping_status=str(row['mapping_status']),
        manufacturer_count=int(row['manufacturer_count']),
        single_manufacturer_flag=int(row['single_manufacturer_flag']),
        historical_shortage_count=int(row['historical_shortage_count']),
        relationship_counts_by_type=rel_counts
    )

def get_drug_relationships(
    node_id: str,
    relationship_type: Optional[str] = None
) -> DrugRelationshipsResponse:
    """
    Retrieve relationship channels connected to a drug node.
    Preserves multi-relationship edge channels for identical node pairs.
    """
    row_matches = state.nodes_df[state.nodes_df['node_id'] == node_id]
    if row_matches.empty:
        raise HTTPException(status_code=404, detail=f"Canonical drug node '{node_id}' not found.")
        
    drug_name = row_matches.iloc[0]['cms_entity_name']
    adj_edges = state.simulator.adjacency[node_id]
    
    if relationship_type:
        rel_type_clean = relationship_type.strip().upper()
        if rel_type_clean not in ['CO_FORMULATED_WITH', 'SAME_THERAPEUTIC_CATEGORY']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid relationship_type '{relationship_type}'. Allowed: CO_FORMULATED_WITH, SAME_THERAPEUTIC_CATEGORY"
            )
        adj_edges = [e for e in adj_edges if e['type'] == rel_type_clean]
        
    items = []
    for edge in adj_edges:
        target_id = edge['target']
        target_info = state.simulator.nodes_dict[target_id]
        
        items.append(DrugRelationshipItem(
            related_node_id=target_id,
            drug_name=target_info['cms_entity_name'],
            relationship_type=edge['type'],
            base_risk_score=target_info['base_risk_score']
        ))
        
    return DrugRelationshipsResponse(
        node_id=node_id,
        drug_name=drug_name,
        total_relationships=len(items),
        relationships=items
    )
