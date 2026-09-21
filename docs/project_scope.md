# Project Scope & Specifications

## Project
**PharmaShortage Predictor** — Pharmaceutical Shortage Risk Analysis and Cascade Simulation.

## Problem Statement
Pharmaceutical shortages can affect medicine availability and may create additional inventory pressure on related drug products.

## Project Goals

### Core Goal
Use real public pharmaceutical data to investigate whether meaningful drug-shortage risk patterns can be identified using data analysis and machine learning.

### Secondary Goal
Build a scenario-based cascade simulator that explores how hypothetical demand redistribution could affect related drug products after a shortage.

## Planned Data Sources
- **FDA Drug Shortages**: Historical and active shortage records and status tracking.
- **CMS Medicare Part D**: Aggregated prescribing data, volume, and utilization metrics.
- **RxNorm**: Standardized nomenclature and semantic relationships for clinical drugs.
- **FDA Orange Book**: Therapeutic equivalence evaluations (may be investigated if useful).

## Important Scientific Boundaries & Limitations
- Do not claim that CMS provides daily or weekly pharmacy demand data.
- Do not claim 7-day demand forecasting unless a suitable temporal dataset is later obtained.
- RxNorm relationships must not be interpreted automatically as therapeutic substitution.
- The system must not recommend medication substitutions to patients.
- Cascade demand redistribution is a scenario simulation unless real substitution-behavior data becomes available.
- Do not claim the ML model is feasible until the Phase 2 data audit is completed.

## Phase 1 Success Criteria
The project problem, objectives, limitations, data assumptions, and feasibility criteria must be documented before ML development begins.

## Phase 2 Feasibility Gate
Before building models, we must:
1. Inspect the exact FDA, CMS, and RxNorm schemas.
2. Determine available identifiers such as drug names, NDCs, and RxCUIs.
3. Prototype entity resolution between the datasets.
4. Measure mapping coverage.
5. Count usable shortage-positive and non-shortage examples.
6. Inspect temporal resolution and available features.
7. Decide whether there is enough data for meaningful ML.
