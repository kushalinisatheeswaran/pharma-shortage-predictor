"""
Comprehensive API Unit & Integration Test Suite for FastAPI Backend
"""

import sys
sys.path.insert(0, ".")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from fastapi.testclient import TestClient
from backend.main import app
from backend.services.state import state
from src.model_inference import predict_risk
from backend.services.drug_service import get_latest_feature_record_for_drug

# Ensure state artifacts are loaded for tests
state.load_artifacts()

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["graph_loaded"] is True

def test_summary_endpoint():
    response = client.get("/api/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_drugs"] == 484
    assert data["relationship_channels"] == 577
    assert data["model_calibrated"] is False

def test_drugs_endpoint_and_filters():
    # Base query
    res = client.get("/api/drugs?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 484
    assert len(data["drugs"]) == 10

    # Search filter
    res_search = client.get("/api/drugs?search=amoxicillin")
    assert res_search.status_code == 200
    data_search = res_search.json()
    assert data_search["total"] > 0
    assert "amoxicillin" in data_search["drugs"][0]["drug_name"].lower()

    # Category filter
    res_cat = client.get("/api/drugs?category=Anti-Infective")
    assert res_cat.status_code == 200
    data_cat = res_cat.json()
    assert data_cat["total"] > 0

def test_drug_detail_endpoint():
    # Known drug
    res = client.get("/api/drugs/NODE_AMOXICILLIN")
    assert res.status_code == 200
    data = res.json()
    assert data["node_id"] == "NODE_AMOXICILLIN"
    assert data["drug_name"] == "AMOXICILLIN"

    # Unknown drug
    res_404 = client.get("/api/drugs/NODE_NONEXISTENT_DRUG")
    assert res_404.status_code == 404

def test_real_ml_inference_endpoint_and_direct_consistency():
    node_id = "NODE_AMOXICILLIN"
    
    # 1. API endpoint call
    res = client.get(f"/api/risk/{node_id}")
    assert res.status_code == 200
    api_data = res.json()
    api_score = api_data["base_risk_score"]
    
    # 2. Direct model_inference.py call
    feat_rec, _ = get_latest_feature_record_for_drug("AMOXICILLIN")
    direct_score = predict_risk(feat_rec)
    
    # 3. Explicit consistency check proving API uses real packaged ML model
    assert abs(api_score - direct_score) < 1e-7
    assert 0.0 <= api_score <= 1.0
    assert "Uncalibrated relative shortage-risk score" in api_data["score_interpretation"]

def test_model_info_endpoint():
    res = client.get("/api/model/info")
    assert res.status_code == 200
    data = res.json()
    assert data["model_version"] == "1.0.0"
    assert data["calibrated"] is False

def test_relationships_endpoint():
    node_id = "NODE_HYDROCHLOROTHIAZIDE"
    
    res = client.get(f"/api/drugs/{node_id}/relationships")
    assert res.status_code == 200
    data = res.json()
    assert data["total_relationships"] == 35

    # Filter by type
    res_co = client.get(f"/api/drugs/{node_id}/relationships?relationship_type=CO_FORMULATED_WITH")
    assert res_co.status_code == 200
    data_co = res_co.json()
    assert data_co["total_relationships"] == 24

def test_graph_summary_endpoint():
    res = client.get("/api/graph/summary")
    assert res.status_code == 200
    data = res.json()
    assert data["nodes"] == 484
    assert data["relationship_channels"] == 577
    assert data["coformulated_relationships"] == 311
    assert data["therapeutic_category_relationships"] == 266

def test_scenario_simulation_endpoint():
    # Valid scenario
    req_payload = {
        "source_node": "NODE_HYDROCHLOROTHIAZIDE",
        "initial_stress": 1.0,
        "coformulation_weight": 0.3,
        "category_weight": 0.1,
        "decay_factor": 0.5,
        "max_hops": 2,
        "minimum_pressure_threshold": 0.01
    }
    res = client.post("/api/scenarios/simulate", json=req_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["summary"]["affected_nodes"] == 16
    assert "Hypothetical inventory stress propagation" in data["interpretation"]
    
    # Verify score bounds
    for item in data["affected_nodes"]:
        assert 0.0 <= item["scenario_pressure"] <= 1.0
        assert 0.0 <= item["scenario_risk_score"] <= 1.0

def test_isolated_node_scenario():
    # Cephalexin is isolated (degree 0)
    req_payload = {
        "source_node": "NODE_CEPHALEXIN",
        "initial_stress": 1.0,
        "coformulation_weight": 0.3,
        "category_weight": 0.1,
        "decay_factor": 0.5,
        "max_hops": 2,
        "minimum_pressure_threshold": 0.01
    }
    res = client.post("/api/scenarios/simulate", json=req_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["summary"]["affected_nodes"] == 0

def test_zero_relationship_weights_scenario():
    req_payload = {
        "source_node": "NODE_HYDROCHLOROTHIAZIDE",
        "initial_stress": 1.0,
        "coformulation_weight": 0.0,
        "category_weight": 0.0,
        "decay_factor": 0.5,
        "max_hops": 2,
        "minimum_pressure_threshold": 0.01
    }
    res = client.post("/api/scenarios/simulate", json=req_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["summary"]["affected_nodes"] == 0

def test_scenario_validation_errors():
    # Invalid source node
    res_404 = client.post("/api/scenarios/simulate", json={
        "source_node": "NODE_INVALID",
        "initial_stress": 1.0
    })
    assert res_404.status_code == 404

    # Invalid initial stress (> 1.0)
    res_422_stress = client.post("/api/scenarios/simulate", json={
        "source_node": "NODE_HYDROCHLOROTHIAZIDE",
        "initial_stress": 1.5
    })
    assert res_422_stress.status_code == 422

    # Invalid max_hops (> 5)
    res_422_hops = client.post("/api/scenarios/simulate", json={
        "source_node": "NODE_HYDROCHLOROTHIAZIDE",
        "max_hops": 10
    })
    assert res_422_hops.status_code == 422

def main():
    print("Executing backend API unit & integration tests...")
    test_health_endpoint()
    print("[OK] Health endpoint passed")
    test_summary_endpoint()
    print("[OK] Summary endpoint passed")
    test_drugs_endpoint_and_filters()
    print("[OK] Drugs endpoint and filters passed")
    test_drug_detail_endpoint()
    print("[OK] Drug detail endpoint passed")
    test_real_ml_inference_endpoint_and_direct_consistency()
    print("[OK] Real ML inference endpoint & consistency test passed")
    test_model_info_endpoint()
    print("[OK] Model info endpoint passed")
    test_relationships_endpoint()
    print("[OK] Relationships endpoint passed")
    test_graph_summary_endpoint()
    print("[OK] Graph summary endpoint passed")
    test_scenario_simulation_endpoint()
    print("[OK] Scenario simulation endpoint passed")
    test_isolated_node_scenario()
    print("[OK] Isolated node scenario passed")
    test_zero_relationship_weights_scenario()
    print("[OK] Zero relationship weights scenario passed")
    test_scenario_validation_errors()
    print("[OK] Scenario validation error handlers passed")
    print("\nAll 12 API unit and integration tests passed successfully!")

if __name__ == "__main__":
    main()
