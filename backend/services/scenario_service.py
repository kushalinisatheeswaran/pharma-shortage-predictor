"""
Scenario Cascade Simulation Service Module
Reuses Phase 7B controlled frontier propagation engine.
"""

from fastapi import HTTPException
import numpy as np
import pandas as pd

from backend.services.state import state
from backend.schemas.scenario import (
    ScenarioSimulateRequest, ScenarioSimulateResponse,
    SourceNodeInfo, ScenarioSummary, ScenarioNodeImpact
)

def run_scenario_simulation(req: ScenarioSimulateRequest) -> ScenarioSimulateResponse:
    """
    Execute scenario stress propagation on Graph B topology.
    """
    source_nid = req.source_node.strip()
    if source_nid not in state.simulator.nodes_dict:
        raise HTTPException(
            status_code=404,
            detail=f"Source node '{source_nid}' not found in canonical drug graph."
        )
        
    # Execute Phase 7B simulator engine
    res_df = state.simulator.run_simulation(
        source_node=source_nid,
        initial_stress=req.initial_stress,
        coformulation_weight=req.coformulation_weight,
        category_weight=req.category_weight,
        decay_factor=req.decay_factor,
        max_hops=req.max_hops,
        minimum_pressure_threshold=req.minimum_pressure_threshold
    )
    
    # Source node info
    source_info = state.simulator.nodes_dict[source_nid]
    source_data = SourceNodeInfo(
        node_id=source_nid,
        drug_name=source_info['cms_entity_name'],
        base_risk_score=source_info['base_risk_score']
    )
    
    # Filter impacted nodes (scenario_pressure > 0)
    impacted_df = res_df[res_df['scenario_pressure'] > 0.0].copy()
    downstream_df = res_df[(res_df['scenario_pressure'] > 0.0) & (res_df['node_id'] != source_nid)]
    
    mean_downstream_p = float(downstream_df['scenario_pressure'].mean()) if len(downstream_df) > 0 else 0.0
    max_downstream_p = float(downstream_df['scenario_pressure'].max()) if len(downstream_df) > 0 else 0.0
    
    summary_data = ScenarioSummary(
        affected_nodes=len(downstream_df),
        mean_downstream_pressure=mean_downstream_p,
        max_downstream_pressure=max_downstream_p
    )
    
    # Sort impacted nodes deterministically: scenario_risk_score descending, then node_id ascending
    sorted_impacted = impacted_df.sort_values(
        by=['scenario_risk_score', 'scenario_pressure', 'node_id'],
        ascending=[False, False, True]
    )
    
    impact_items = []
    for _, row in sorted_impacted.iterrows():
        impact_items.append(ScenarioNodeImpact(
            node_id=row['node_id'],
            drug_name=row['cms_entity_name'],
            therapeutic_category=row['therapeutic_category'],
            base_risk_score=float(row['base_risk_score']),
            scenario_pressure=float(row['scenario_pressure']),
            scenario_risk_score=float(row['scenario_risk_score']),
            hop_distance=int(row['hop_distance']) if pd.notnull(row['hop_distance']) else None
        ))
        
    config_dict = {
        "source_node": source_nid,
        "initial_stress": req.initial_stress,
        "coformulation_weight": req.coformulation_weight,
        "category_weight": req.category_weight,
        "decay_factor": req.decay_factor,
        "max_hops": req.max_hops,
        "minimum_pressure_threshold": req.minimum_pressure_threshold
    }
    
    return ScenarioSimulateResponse(
        scenario_configuration=config_dict,
        source=source_data,
        summary=summary_data,
        affected_nodes=impact_items,
        interpretation="Hypothetical inventory stress propagation across an analytical drug relationship network. Results do not represent clinical substitution, patient switching, observed demand transfer, causal shortage transmission, or calibrated shortage probabilities."
    )
