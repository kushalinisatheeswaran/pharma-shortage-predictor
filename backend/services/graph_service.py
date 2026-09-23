"""
Graph & System Summary Service Module
"""

import networkx as nx
from backend.services.state import state
from backend.schemas.graph import GraphSummaryResponse, SystemSummaryResponse

def get_graph_summary() -> GraphSummaryResponse:
    """
    Calculate Graph B metrics.
    """
    nodes_cnt = len(state.nodes_df)
    sim_edges = state.edges_df[state.edges_df['graph_view'] == 'SIMILARITY']
    rel_channels = len(sim_edges)
    
    # Calculate unique node pairs
    node_pairs = set()
    for _, e in sim_edges.iterrows():
        pair = tuple(sorted([e['source_node'], e['target_node']]))
        node_pairs.add(pair)
    unique_pairs_cnt = len(node_pairs)
    
    degrees = state.simulator.degrees
    connected_cnt = sum(1 for d in degrees.values() if d > 0)
    isolated_cnt = sum(1 for d in degrees.values() if d == 0)
    
    comps = list(nx.connected_components(state.graph_sim))
    comps_cnt = len(comps)
    largest_comp_size = max(len(c) for c in comps) if comps else 0
    
    counts = sim_edges['relationship_type'].value_counts().to_dict()
    co_form_cnt = counts.get('CO_FORMULATED_WITH', 0)
    cat_cnt = counts.get('SAME_THERAPEUTIC_CATEGORY', 0)
    
    return GraphSummaryResponse(
        nodes=nodes_cnt,
        relationship_channels=rel_channels,
        unique_node_pairs=unique_pairs_cnt,
        connected_nodes=connected_cnt,
        isolated_nodes=isolated_cnt,
        connected_components=comps_cnt,
        largest_component_size=largest_comp_size,
        coformulated_relationships=co_form_cnt,
        therapeutic_category_relationships=cat_cnt
    )

def get_system_summary() -> SystemSummaryResponse:
    """
    Calculate system overview summary.
    """
    g_summary = get_graph_summary()
    mapped_cnt = int((state.nodes_df['canonical_rxcui'] != 'UNMAPPED').sum())
    
    return SystemSummaryResponse(
        total_drugs=g_summary.nodes,
        mapped_drugs=mapped_cnt,
        connected_nodes=g_summary.connected_nodes,
        isolated_nodes=g_summary.isolated_nodes,
        relationship_channels=g_summary.relationship_channels,
        unique_node_pairs=g_summary.unique_node_pairs,
        relationship_counts={
            'CO_FORMULATED_WITH': g_summary.coformulated_relationships,
            'SAME_THERAPEUTIC_CATEGORY': g_summary.therapeutic_category_relationships
        },
        model_name=state.metadata["model_name"],
        model_role=state.metadata["model_role"],
        model_calibrated=False
    )
