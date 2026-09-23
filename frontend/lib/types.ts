export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  model_loaded: boolean;
  graph_loaded: boolean;
}

export interface SystemSummary {
  total_drugs: number;
  mapped_drugs: number;
  connected_nodes: number;
  isolated_nodes: number;
  relationship_channels: number;
  unique_node_pairs: number;
  relationship_counts: Record<string, number>;
  model_name: string;
  model_role: string;
  model_calibrated: boolean;
}

export interface DrugSummary {
  node_id: string;
  drug_name: string;
  therapeutic_category: string;
  latest_feature_year: number;
  base_risk_score: number;
  relationship_count: number;
  is_connected: boolean;
}

export interface DrugListResponse {
  total: number;
  limit: number;
  offset: number;
  drugs: DrugSummary[];
}

export interface DrugDetail extends DrugSummary {
  canonical_rxcui: string;
  mapping_status: string;
  manufacturer_count: number;
  single_manufacturer_flag: number;
  historical_shortage_count: number;
  relationship_counts_by_type: Record<string, number>;
}

export interface DrugRelationshipItem {
  related_node_id: string;
  drug_name: string;
  relationship_type: "CO_FORMULATED_WITH" | "SAME_THERAPEUTIC_CATEGORY" | string;
  base_risk_score: number;
}

export interface DrugRelationshipsResponse {
  node_id: string;
  drug_name: string;
  total_relationships: number;
  relationships: DrugRelationshipItem[];
}

export interface RiskPredictionResponse {
  node_id: string;
  drug_name: string;
  feature_year: number;
  model_name: string;
  base_risk_score: number;
  stored_graph_risk_score: number;
  consistency_status: string;
  score_interpretation: string;
}

export interface ValidationMetrics {
  pr_auc: number;
  roc_auc: number;
  precision: number;
  recall: number;
  f1: number;
  accuracy: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  decision_threshold: number;
}

export interface ModelInfo {
  model_name: string;
  model_version: string;
  model_role: string;
  training_feature_years: number[];
  target_years: number[];
  training_observations: number;
  training_positive_count: number;
  validation_metrics: ValidationMetrics;
  calibrated: boolean;
  score_interpretation: string;
}

export interface GraphSummary {
  nodes: number;
  relationship_channels: number;
  unique_node_pairs: number;
  connected_nodes: number;
  isolated_nodes: number;
  connected_components: number;
  largest_component_size: number;
  coformulated_relationships: number;
  therapeutic_category_relationships: number;
}

export interface ScenarioSimulateRequest {
  source_node: string;
  initial_stress: number;
  coformulation_weight: number;
  category_weight: number;
  decay_factor: number;
  max_hops: number;
  minimum_pressure_threshold: number;
}

export interface SourceNodeInfo {
  node_id: string;
  drug_name: string;
  base_risk_score: number;
}

export interface ScenarioSummary {
  affected_nodes: number;
  mean_downstream_pressure: number;
  max_downstream_pressure: number;
}

export interface ScenarioNodeImpact {
  node_id: string;
  drug_name: string;
  therapeutic_category: string;
  base_risk_score: number;
  scenario_pressure: number;
  scenario_risk_score: number;
  hop_distance?: number | null;
}

export interface ScenarioSimulateResponse {
  scenario_configuration: Record<string, unknown>;
  source: SourceNodeInfo;
  summary: ScenarioSummary;
  affected_nodes: ScenarioNodeImpact[];
  interpretation: string;
}
