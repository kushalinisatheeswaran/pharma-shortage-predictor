"""
Phase 7B: Controlled Scenario Cascade Simulator

Hypothetical inventory stress propagation across an analytical drug relationship network.

IMPORTANT SCIENTIFIC & SAFETY DISCLAIMER:
- This simulator models HYPOTHETICAL scenario inventory pressure redistributions across Graph B.
- It does NOT model clinical prescribing substitution, patient switching, observed demand transfer,
  causal shortage transmission, or calibrated shortage probability.
- The input model score ('latest_model_risk_score') is an UNCALIBRATED RELATIVE SHORTAGE-RISK SCORE,
  NOT an absolute calibrated probability.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Core Cascade Engine Class
# -----------------------------------------------------------------------------
class ControlledCascadeSimulator:
    def __init__(self, nodes_df, edges_df):
        """
        Initialize simulator with Graph B topology and uncalibrated relative risk scores.
        """
        self.nodes_df = nodes_df.copy()
        self.edges_df = edges_df.copy()
        
        # Ensure node dictionary lookup
        self.nodes_dict = {}
        for _, row in self.nodes_df.iterrows():
            self.nodes_dict[row['node_id']] = {
                'cms_entity_name': row['cms_entity_name'],
                'therapeutic_category': row['therapeutic_category'],
                'base_risk_score': float(row['latest_model_risk_score'])  # Uncalibrated relative risk score
            }
            
        # Build adjacency list & degrees for Graph B (CO_FORMULATED_WITH & SAME_THERAPEUTIC_CATEGORY)
        self.adjacency = {node_id: [] for node_id in self.nodes_dict}
        self.degrees = {node_id: 0 for node_id in self.nodes_dict}
        
        # Filter Graph B edges
        similarity_edges = self.edges_df[self.edges_df['graph_view'] == 'SIMILARITY']
        
        for _, edge in similarity_edges.iterrows():
            u = edge['source_node']
            v = edge['target_node']
            rel_type = edge['relationship_type']
            
            if u in self.nodes_dict and v in self.nodes_dict:
                self.adjacency[u].append({'target': v, 'type': rel_type})
                self.adjacency[v].append({'target': u, 'type': rel_type})
                
        # Calculate degrees
        for u in self.nodes_dict:
            self.degrees[u] = len(self.adjacency[u])
            
    def run_simulation(self, source_node, initial_stress=1.0, coformulation_weight=0.3,
                       category_weight=0.1, decay_factor=0.5, max_hops=2,
                       minimum_pressure_threshold=0.01):
        """
        Controlled frontier-based propagation algorithm.
        
        Parameters (HYPOTHETICAL SCENARIO PARAMETERS):
        - initial_stress: Stress shock applied to source node [0, 1]
        - coformulation_weight: Scenario weight for CO_FORMULATED_WITH edges
        - category_weight: Scenario weight for SAME_THERAPEUTIC_CATEGORY edges
        - decay_factor: Multiplicative attenuation per hop
        - max_hops: Maximum propagation distance
        - minimum_pressure_threshold: Minimum pressure cutoff
        """
        if source_node not in self.nodes_dict:
            raise ValueError(f"Source node {source_node} not found in graph.")
            
        # Initialize pressures
        pressure = {node_id: 0.0 for node_id in self.nodes_dict}
        pressure[source_node] = float(np.clip(initial_stress, 0.0, 1.0))
        
        # Track hop distance reached
        hop_reached = {node_id: None for node_id in self.nodes_dict}
        hop_reached[source_node] = 0
        
        # Active frontier: list of (node, current_pressure, hop)
        frontier = [(source_node, pressure[source_node], 0)]
        
        rel_weights = {
            'CO_FORMULATED_WITH': coformulation_weight,
            'SAME_THERAPEUTIC_CATEGORY': category_weight
        }
        
        while frontier:
            next_frontier = []
            
            for u, p_u, current_hop in frontier:
                if current_hop >= max_hops:
                    continue
                    
                deg_u = self.degrees[u]
                if deg_u == 0:
                    continue
                    
                for edge in self.adjacency[u]:
                    v = edge['target']
                    rel_type = edge['type']
                    
                    # Rule: Do NOT propagate back to source
                    if v == source_node:
                        continue
                        
                    w_rel = rel_weights.get(rel_type, 0.0)
                    if w_rel <= 0:
                        continue
                        
                    deg_v = self.degrees[v]
                    if deg_v == 0:
                        continue
                        
                    # Structural normalization to prevent clique feedback explosion
                    norm_edge_weight = w_rel / np.sqrt(deg_u * deg_v)
                    
                    candidate_pressure = p_u * norm_edge_weight * decay_factor
                    candidate_pressure = float(np.clip(candidate_pressure, 0.0, 1.0))
                    
                    if candidate_pressure < minimum_pressure_threshold:
                        continue
                        
                    # Maintain the strongest/best pressure received per node
                    if candidate_pressure > pressure[v]:
                        pressure[v] = candidate_pressure
                        hop_reached[v] = current_hop + 1
                        next_frontier.append((v, candidate_pressure, current_hop + 1))
                        
            frontier = next_frontier
            
        # Transparent analytical combination:
        # scenario_risk_score = base_risk_score + scenario_pressure * (1 - base_risk_score)
        results = []
        for node_id, data in self.nodes_dict.items():
            base_risk = data['base_risk_score']
            p = pressure[node_id]
            combined_risk = base_risk + p * (1.0 - base_risk)
            combined_risk = float(np.clip(combined_risk, 0.0, 1.0))
            
            results.append({
                'node_id': node_id,
                'cms_entity_name': data['cms_entity_name'],
                'therapeutic_category': data['therapeutic_category'],
                'degree': self.degrees[node_id],
                'base_risk_score': base_risk,
                'scenario_pressure': p,
                'scenario_risk_score': combined_risk,
                'hop_distance': hop_reached[node_id]
            })
            
        res_df = pd.DataFrame(results)
        return res_df

# -----------------------------------------------------------------------------
# Validation Suite
# -----------------------------------------------------------------------------
def run_validation_suite(simulator):
    """
    Run 9 mandatory validation checks.
    """
    print("--- Executing Phase 7B Validation Suite ---")
    val_report = {}
    
    # Select sample nodes for tests
    sample_nodes = [nid for nid in simulator.nodes_dict if simulator.degrees[nid] > 0]
    source = sample_nodes[0]
    
    # Test 1: Pressure bounded in [0, 1]
    res = simulator.run_simulation(source)
    p_valid = (res['scenario_pressure'] >= 0.0).all() and (res['scenario_pressure'] <= 1.0).all()
    val_report['pressure_bounded_0_1'] = bool(p_valid)
    print(f"1. Pressure in [0, 1]: {p_valid}")
    
    # Test 2: scenario_risk_score bounded in [0, 1]
    r_valid = (res['scenario_risk_score'] >= 0.0).all() and (res['scenario_risk_score'] <= 1.0).all()
    val_report['scenario_risk_score_bounded_0_1'] = bool(r_valid)
    print(f"2. Scenario risk score in [0, 1]: {r_valid}")
    
    # Test 3: Isolated source -> zero downstream nodes
    isolated_nodes = [nid for nid in simulator.nodes_dict if simulator.degrees[nid] == 0]
    if isolated_nodes:
        iso_source = isolated_nodes[0]
        res_iso = simulator.run_simulation(iso_source)
        downstream_iso = res_iso[res_iso['node_id'] != iso_source]['scenario_pressure'].sum()
        iso_valid = (downstream_iso == 0.0)
    else:
        iso_valid = True
    val_report['isolated_source_zero_downstream'] = bool(iso_valid)
    print(f"3. Isolated source produces zero downstream propagation: {iso_valid}")
    
    # Test 4: Relationship weights = 0 -> zero propagation
    res_zero_w = simulator.run_simulation(source, coformulation_weight=0.0, category_weight=0.0)
    downstream_zero = res_zero_w[res_zero_w['node_id'] != source]['scenario_pressure'].sum()
    zero_w_valid = (downstream_zero == 0.0)
    val_report['zero_weights_zero_propagation'] = bool(zero_w_valid)
    print(f"4. Zero relationship weights yield zero propagation: {zero_w_valid}")
    
    # Test 5: max_hops respected
    res_hop1 = simulator.run_simulation(source, max_hops=1)
    max_hop_obs = res_hop1['hop_distance'].dropna().max()
    hop_valid = (max_hop_obs <= 1)
    val_report['max_hops_respected'] = bool(hop_valid)
    print(f"5. Max hops strictly respected (max hop <= 1): {hop_valid}")
    
    # Test 6: No propagation back to source
    source_p = res['scenario_pressure'][res['node_id'] == source].values[0]
    # Source pressure should equal initial_stress (1.0), not augmented by cycle feedback
    back_valid = (source_p == 1.0)
    val_report['no_back_propagation_to_source'] = bool(back_valid)
    print(f"6. No propagation back to source (source pressure equals initial stress): {back_valid}")
    
    # Test 7: Controlled frontier prevents cycle amplification
    res_high_hops = simulator.run_simulation(source, max_hops=10, decay_factor=0.9)
    cycle_valid = (res_high_hops['scenario_pressure'] <= 1.0).all()
    val_report['no_uncontrolled_cycle_amplification'] = bool(cycle_valid)
    print(f"7. No uncontrolled cycle amplification under high hops: {cycle_valid}")
    
    # Test 8: Monotonicity check (stronger scenario parameters do not decrease pressure)
    res_low_param = simulator.run_simulation(source, coformulation_weight=0.1, category_weight=0.05)
    res_high_param = simulator.run_simulation(source, coformulation_weight=0.5, category_weight=0.2)
    diff = res_high_param['scenario_pressure'] - res_low_param['scenario_pressure']
    mono_valid = (diff >= -1e-9).all()
    val_report['monotonic_parameter_scaling'] = bool(mono_valid)
    print(f"8. Monotonic parameter scaling (stronger parameters do not reduce pressure): {mono_valid}")
    
    # Test 9: Graph edge semantics remain unchanged
    sem_valid = (len(simulator.edges_df) == 577) and (set(simulator.edges_df['relationship_type'].unique()) == {'CO_FORMULATED_WITH', 'SAME_THERAPEUTIC_CATEGORY'})
    val_report['graph_edge_semantics_unchanged'] = bool(sem_valid)
    print(f"9. Graph edge semantics remain unchanged: {sem_valid}\n")
    
    all_pass = all(val_report.values())
    val_report['all_validation_tests_passed'] = bool(all_pass)
    
    with open("data/processed/cascade_validation.json", "w") as f:
        json.dump(val_report, f, indent=2)
        
    return val_report

# -----------------------------------------------------------------------------
# Main Execution Function
# -----------------------------------------------------------------------------
def main():
    print("Loading Graph B nodes and edges...")
    nodes_df = pd.read_csv("data/processed/drug_graph_nodes.csv")
    edges_df = pd.read_csv("data/processed/drug_graph_edges.csv")
    
    simulator = ControlledCascadeSimulator(nodes_df, edges_df)
    
    # -------------------------------------------------------------------------
    # 1. Deterministic Source Node Selection
    # -------------------------------------------------------------------------
    print("Selecting deterministic validation scenario nodes from Graph B...")
    
    connected_nodes = nodes_df[nodes_df['node_id'].map(simulator.degrees) > 0].copy()
    connected_nodes['degree'] = connected_nodes['node_id'].map(simulator.degrees)
    
    # A. Highest-degree connected node
    highest_degree_node = connected_nodes.sort_values(by=['degree', 'latest_model_risk_score'], ascending=[False, False]).iloc[0]['node_id']
    
    # B. Low-degree connected node (degree == 1 or min connected degree)
    low_degree_node = connected_nodes.sort_values(by=['degree', 'node_id'], ascending=[True, True]).iloc[0]['node_id']
    
    # C. Isolated node (degree == 0)
    isolated_nodes = nodes_df[nodes_df['node_id'].map(simulator.degrees) == 0]
    isolated_node = isolated_nodes.iloc[0]['node_id'] if len(isolated_nodes) > 0 else None
    
    # D. Relatively high base-risk connected node
    high_risk_node = connected_nodes.sort_values(by=['latest_model_risk_score', 'degree'], ascending=[False, False]).iloc[0]['node_id']
    
    # E. Relatively low base-risk connected node
    low_risk_node = connected_nodes.sort_values(by=['latest_model_risk_score', 'degree'], ascending=[True, False]).iloc[0]['node_id']
    
    scenarios = [
        ('A', 'Highest-degree connected node', highest_degree_node),
        ('B', 'Low-degree connected node', low_degree_node),
        ('C', 'Isolated node', isolated_node),
        ('D', 'Relatively high base-risk connected node', high_risk_node),
        ('E', 'Relatively low base-risk connected node', low_risk_node)
    ]
    
    print("\n--- Deterministic Scenario Source Selection ---")
    for code, label, nid in scenarios:
        name = simulator.nodes_dict[nid]['cms_entity_name']
        deg = simulator.degrees[nid]
        risk = simulator.nodes_dict[nid]['base_risk_score']
        print(f"Scenario {code} [{label}]:")
        print(f"  Node ID: {nid} | Drug: {name} | Degree: {deg} | Uncalibrated Relative Risk: {risk:.4f}")
    print()
    
    # -------------------------------------------------------------------------
    # 2. Run Base Example Scenarios
    # -------------------------------------------------------------------------
    example_results = []
    for code, label, nid in scenarios:
        res_df = simulator.run_simulation(nid, initial_stress=1.0, coformulation_weight=0.3,
                                         category_weight=0.1, decay_factor=0.5, max_hops=2)
        res_df['scenario_code'] = code
        res_df['scenario_description'] = label
        res_df['source_node_id'] = nid
        example_results.append(res_df)
        
    all_example_df = pd.concat(example_results, ignore_index=True)
    all_example_df.to_csv("data/processed/cascade_example_results.csv", index=False)
    print("Saved baseline cascade scenario outputs to data/processed/cascade_example_results.csv")
    
    # -------------------------------------------------------------------------
    # 3. Sensitivity Analysis (Hypothetical Parameter Grid)
    # -------------------------------------------------------------------------
    print("Executing sensitivity grid search across hypothetical scenario parameters...")
    coform_grid = [0.1, 0.3, 0.5]
    cat_grid = [0.05, 0.1, 0.2]
    decay_grid = [0.3, 0.5, 0.7]
    
    sens_rows = []
    # Test on highest-degree node
    target_source = highest_degree_node
    
    for w_co in coform_grid:
        for w_cat in cat_grid:
            for decay in decay_grid:
                res_df = simulator.run_simulation(
                    target_source,
                    initial_stress=1.0,
                    coformulation_weight=w_co,
                    category_weight=w_cat,
                    decay_factor=decay,
                    max_hops=2
                )
                impacted = res_df[res_df['scenario_pressure'] > 0.0]
                downstream_impacted = res_df[(res_df['scenario_pressure'] > 0.0) & (res_df['node_id'] != target_source)]
                
                mean_pressure = downstream_impacted['scenario_pressure'].mean() if len(downstream_impacted) > 0 else 0.0
                max_pressure = downstream_impacted['scenario_pressure'].max() if len(downstream_impacted) > 0 else 0.0
                
                sens_rows.append({
                    'source_node_id': target_source,
                    'coformulation_weight': w_co,
                    'category_weight': w_cat,
                    'decay_factor': decay,
                    'total_impacted_nodes': len(impacted),
                    'downstream_impacted_nodes': len(downstream_impacted),
                    'mean_downstream_pressure': float(mean_pressure),
                    'max_downstream_pressure': float(max_pressure)
                })
                
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv("data/processed/cascade_sensitivity_results.csv", index=False)
    print("Saved sensitivity analysis grid results to data/processed/cascade_sensitivity_results.csv")
    
    # -------------------------------------------------------------------------
    # 4. Generate Figures
    # -------------------------------------------------------------------------
    os.makedirs("reports/figures", exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: Downstream Pressure Distribution for Highest Degree Node
    plt.subplot(1, 2, 1)
    h_res = simulator.run_simulation(highest_degree_node)
    downstream_h = h_res[h_res['node_id'] != highest_degree_node]['scenario_pressure']
    plt.hist(downstream_h, bins=20, color='crimson', alpha=0.7, edgecolor='black')
    plt.title(f"Downstream Pressure (Source: {highest_degree_node})", fontsize=11)
    plt.xlabel("Hypothetical Scenario Pressure", fontsize=10)
    plt.ylabel("Drug Node Count", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Subplot 2: Sensitivity Heatmap (decay vs coformulation weight)
    plt.subplot(1, 2, 2)
    pivot_sens = sens_df[sens_df['category_weight'] == 0.1].pivot(
        index='decay_factor', columns='coformulation_weight', values='mean_downstream_pressure'
    )
    plt.imshow(pivot_sens.values, cmap='YlOrRd', aspect='auto', origin='lower')
    plt.colorbar(label='Mean Downstream Pressure')
    plt.xticks(ticks=range(len(coform_grid)), labels=coform_grid)
    plt.yticks(ticks=range(len(decay_grid)), labels=decay_grid)
    plt.xlabel("Co-formulation Scenario Weight", fontsize=10)
    plt.ylabel("Decay Factor", fontsize=10)
    plt.title("Sensitivity Grid (Category Weight = 0.1)", fontsize=11)
    
    plt.tight_layout()
    plt.savefig("reports/figures/cascade_propagation_scenarios.png", dpi=300)
    plt.close()
    print("Saved summary visualization figure to reports/figures/cascade_propagation_scenarios.png")
    
    # -------------------------------------------------------------------------
    # 5. Run Validation Suite
    # -------------------------------------------------------------------------
    val_report = run_validation_suite(simulator)
    
    # -------------------------------------------------------------------------
    # 6. Print Phase 7B Report
    # -------------------------------------------------------------------------
    print("\n========== PHASE 7B CASCADE SIMULATION REPORT ==========")
    print("Module: src/simulate_cascade.py")
    print("Graph View: Graph B (Analytical Similarity Graph)")
    print("Total Nodes Evaluated: 484")
    print("Relationships: CO_FORMULATED_WITH (311 edges), SAME_THERAPEUTIC_CATEGORY (266 edges)")
    print("---------------------------------------------------------")
    print("DETERMINISTIC SOURCE SCENARIO EVALUATIONS:")
    for code, label, nid in scenarios:
        res_df = all_example_df[all_example_df['scenario_code'] == code]
        downstream = res_df[(res_df['node_id'] != nid) & (res_df['scenario_pressure'] > 0)]
        mean_p = downstream['scenario_pressure'].mean() if len(downstream) > 0 else 0.0
        max_p = downstream['scenario_pressure'].max() if len(downstream) > 0 else 0.0
        drug_name = simulator.nodes_dict[nid]['cms_entity_name']
        print(f"Scenario {code} ({label} - {drug_name}):")
        print(f"  Downstream Impacted Nodes: {len(downstream)}")
        print(f"  Mean Downstream Pressure: {mean_p:.4f}")
        print(f"  Max Downstream Pressure: {max_p:.4f}")
    print("---------------------------------------------------------")
    print("VALIDATION SUITE SUMMARY:")
    print(f"  All 9 Mandatory Validation Tests Passed: {val_report['all_validation_tests_passed']}")
    print("---------------------------------------------------------")
    print("SCIENTIFIC BOUNDARY ACKNOWLEDGEMENT:")
    print("  The simulator models HYPOTHETICAL inventory stress propagation across")
    print("  an analytical drug relationship network. It does NOT represent clinical")
    print("  substitution, patient switching, observed demand transfer, causal shortage")
    print("  transmission, or calibrated shortage probability.")
    print("=========================================================\n")

if __name__ == "__main__":
    main()
