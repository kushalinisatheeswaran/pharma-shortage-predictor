"""
Scenario Stress Cascade Router
"""

from fastapi import APIRouter
from backend.schemas.scenario import ScenarioSimulateRequest, ScenarioSimulateResponse
from backend.services.scenario_service import run_scenario_simulation

router = APIRouter(prefix="/api/scenarios", tags=["Scenario Cascade Simulation"])

@router.post(
    "/simulate",
    response_model=ScenarioSimulateResponse,
    summary="Simulate Hypothetical Inventory Stress Cascade",
    description="Executes the validated Phase 7B controlled frontier-based stress propagation algorithm on Graph B."
)
def simulate_scenario(req: ScenarioSimulateRequest):
    return run_scenario_simulation(req)
