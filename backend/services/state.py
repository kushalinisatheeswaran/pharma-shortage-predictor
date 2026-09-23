"""
Global Application State & Artifact Cache
Loaded ONCE at FastAPI startup during lifespan event.
"""

import os
import json
import pandas as pd
import networkx as nx

from backend.config import settings
from src.model_inference import load_model, get_model_metadata, get_feature_schema
from src.simulate_cascade import ControlledCascadeSimulator

class AppState:
    def __init__(self):
        self.model = None
        self.metadata = None
        self.schema = None
        self.nodes_df = None
        self.edges_df = None
        self.ml_df = None
        self.latest_feature_obs = None
        self.simulator = None
        self.graph_sim = None
        
    def load_artifacts(self):
        print("Loading backend artifacts at startup...")
        
        # Load ML Inference Pipeline & Specs
        self.model = load_model(settings.MODEL_PATH)
        self.metadata = get_model_metadata(settings.METADATA_PATH)
        self.schema = get_feature_schema(settings.SCHEMA_PATH)
        
        # Load Dataframes
        self.nodes_df = pd.read_csv(settings.NODES_PATH)
        self.edges_df = pd.read_csv(settings.EDGES_PATH)
        self.ml_df = pd.read_csv(settings.ML_DATA_PATH)
        
        # Derive latest valid feature observation per drug entity
        # Sort by feature_year_t ascending, pick last row for each cms_entity_name
        self.latest_feature_obs = self.ml_df.sort_values('feature_year_t').groupby('cms_entity_name').last().reset_index()
        
        # Initialize Phase 7B Cascade Simulator
        self.simulator = ControlledCascadeSimulator(self.nodes_df, self.edges_df)
        
        # Build NetworkX simple graph for component analysis
        self.graph_sim = nx.Graph()
        for _, n in self.nodes_df.iterrows():
            self.graph_sim.add_node(n['node_id'])
            
        similarity_edges = self.edges_df[self.edges_df['graph_view'] == 'SIMILARITY']
        for _, e in similarity_edges.iterrows():
            self.graph_sim.add_edge(e['source_node'], e['target_node'])
            
        print(f"Loaded {len(self.nodes_df)} nodes, {len(self.edges_df)} edges, and {len(self.latest_feature_obs)} entity records.")

state = AppState()
