"""
Backend Configuration Module
"""

import os

class Settings:
    PROJECT_TITLE: str = "MedCascade API"
    PROJECT_DESCRIPTION: str = "Predictive Pharmaceutical Shortage & Cascading Demand Intelligence"
    PROJECT_VERSION: str = "0.1.0"
    
    # Allowed CORS Origins
    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    
    # Artifact paths
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/medcascade_logistic_pipeline.joblib")
    METADATA_PATH: str = os.getenv("METADATA_PATH", "models/medcascade_model_metadata.json")
    SCHEMA_PATH: str = os.getenv("SCHEMA_PATH", "models/medcascade_feature_schema.json")
    NODES_PATH: str = os.getenv("NODES_PATH", "data/processed/drug_graph_nodes.csv")
    EDGES_PATH: str = os.getenv("EDGES_PATH", "data/processed/drug_graph_edges.csv")
    ML_DATA_PATH: str = os.getenv("ML_DATA_PATH", "data/processed/ml_feature_candidates.csv")

settings = Settings()
