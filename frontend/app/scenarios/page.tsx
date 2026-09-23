"use client";

import { useEffect, useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { 
  Network, 
  Play, 
  RotateCcw, 
  Info
} from "lucide-react";
import { api } from "@/lib/api";
import { DrugSummary, ScenarioSimulateRequest, ScenarioSimulateResponse } from "@/lib/types";
import { LoadingState } from "@/components/common/LoadingState";
import { ErrorState } from "@/components/common/ErrorState";
import { DisclaimerAlert } from "@/components/common/DisclaimerAlert";
import { formatScore } from "@/lib/format";

function ScenarioLabContent() {
  const searchParams = useSearchParams();
  const defaultSource = searchParams.get("source") || "NODE_HYDROCHLOROTHIAZIDE";

  const [drugs, setDrugs] = useState<DrugSummary[]>([]);
  const [sourceNode, setSourceNode] = useState<string>(defaultSource);
  
  // Hypothetical scenario parameters
  const [initialStress, setInitialStress] = useState<number>(1.0);
  const [coformulationWeight, setCoformulationWeight] = useState<number>(0.3);
  const [categoryWeight, setCategoryWeight] = useState<number>(0.1);
  const [decayFactor, setDecayFactor] = useState<number>(0.5);
  const [maxHops, setMaxHops] = useState<number>(2);
  const [minThreshold, setMinThreshold] = useState<number>(0.01);

  const [simulationResult, setSimulationResult] = useState<ScenarioSimulateResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [drugsLoading, setDrugsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load connected drugs for dropdown selection
  useEffect(() => {
    api
      .getDrugs({ connected_only: true, limit: 300 })
      .then((res) => {
        setDrugs(res.drugs);
      })
      .catch((err) => {
        console.error("Failed to load drug options", err);
      })
      .finally(() => setDrugsLoading(false));
  }, []);

  const handleManualRun = () => {
    setLoading(true);
    setError(null);
    const payload: ScenarioSimulateRequest = {
      source_node: sourceNode,
      initial_stress: initialStress,
      coformulation_weight: coformulationWeight,
      category_weight: categoryWeight,
      decay_factor: decayFactor,
      max_hops: maxHops,
      minimum_pressure_threshold: minThreshold,
    };

    api
      .simulateScenario(payload)
      .then((res) => {
        setSimulationResult(res);
      })
      .catch((err) => {
        setError(err.message || "Failed to execute scenario simulation.");
      })
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    let ignore = false;
    const payload: ScenarioSimulateRequest = {
      source_node: sourceNode,
      initial_stress: initialStress,
      coformulation_weight: coformulationWeight,
      category_weight: categoryWeight,
      decay_factor: decayFactor,
      max_hops: maxHops,
      minimum_pressure_threshold: minThreshold,
    };

    api
      .simulateScenario(payload)
      .then((res) => {
        if (!ignore) {
          setSimulationResult(res);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!ignore) {
          setError(err.message || "Failed to execute scenario simulation.");
          setLoading(false);
        }
      });

    return () => {
      ignore = true;
    };
  }, [sourceNode, initialStress, coformulationWeight, categoryWeight, decayFactor, maxHops, minThreshold]);

  const resetDefaults = () => {
    setInitialStress(1.0);
    setCoformulationWeight(0.3);
    setCategoryWeight(0.1);
    setDecayFactor(0.5);
    setMaxHops(2);
    setMinThreshold(0.01);
  };

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 md:p-8 space-y-3">
        <div className="inline-flex items-center space-x-2 text-xs font-semibold uppercase tracking-wider text-sky-400 bg-sky-500/10 px-3 py-1 rounded-full border border-sky-500/20">
          <Network className="w-3.5 h-3.5" />
          <span>Scenario Stress Propagation Lab</span>
        </div>
        <h1 className="text-2xl md:text-3xl font-extrabold text-slate-100 tracking-tight">
          Scenario Lab
        </h1>
        <p className="text-sm text-slate-300 max-w-3xl leading-relaxed">
          Explore hypothetical inventory stress cascades across Graph B network relationships using degree-normalized controlled frontier propagation.
        </p>
      </div>

      {/* Grid: Controls + Results */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

        {/* Left Column: Hypothetical Scenario Parameter Controls */}
        <div className="lg:col-span-4 bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-5 h-fit">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h2 className="text-base font-bold text-slate-100 tracking-tight">
              Hypothetical Parameters
            </h2>
            <button
              onClick={resetDefaults}
              className="text-xs text-slate-400 hover:text-slate-200 flex items-center space-x-1"
              title="Reset parameters to validated baseline defaults"
            >
              <RotateCcw className="w-3 h-3" />
              <span>Reset</span>
            </button>
          </div>

          {/* Source Drug Selector */}
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-slate-300 flex items-center justify-between">
              <span>Source Drug Node</span>
              <span title="Target node receiving initial scenario stress shock">
                <Info className="w-3.5 h-3.5 text-slate-500" />
              </span>
            </label>
            {drugsLoading ? (
              <div className="text-xs text-slate-500">Loading drug options...</div>
            ) : (
              <select
                value={sourceNode}
                onChange={(e) => setSourceNode(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-200 focus:outline-none focus:border-sky-500/50"
              >
                {drugs.map((d) => (
                  <option key={d.node_id} value={d.node_id}>
                    {d.drug_name} ({d.node_id} - Deg: {d.relationship_count})
                  </option>
                ))}
              </select>
            )}
          </div>

          {/* Initial Stress Slider */}
          <div className="space-y-1.5">
            <div className="flex justify-between text-xs">
              <span className="font-medium text-slate-300">Initial Stress Shock</span>
              <span className="font-mono text-sky-400">{initialStress.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={initialStress}
              onChange={(e) => setInitialStress(parseFloat(e.target.value))}
              className="w-full accent-sky-400 bg-slate-950"
            />
            <p className="text-[10px] text-slate-500">Starting disruption intensity assigned to source node.</p>
          </div>

          {/* Co-formulation Weight */}
          <div className="space-y-1.5">
            <div className="flex justify-between text-xs">
              <span className="font-medium text-slate-300">Co-formulation Weight</span>
              <span className="font-mono text-indigo-400">{coformulationWeight.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={coformulationWeight}
              onChange={(e) => setCoformulationWeight(parseFloat(e.target.value))}
              className="w-full accent-indigo-400 bg-slate-950"
            />
            <p className="text-[10px] text-slate-500">Propagation strength across co-formulated edges.</p>
          </div>

          {/* Category Weight */}
          <div className="space-y-1.5">
            <div className="flex justify-between text-xs">
              <span className="font-medium text-slate-300">Therapeutic Category Weight</span>
              <span className="font-mono text-emerald-400">{categoryWeight.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={categoryWeight}
              onChange={(e) => setCategoryWeight(parseFloat(e.target.value))}
              className="w-full accent-emerald-400 bg-slate-950"
            />
            <p className="text-[10px] text-slate-500">Propagation strength across same category edges.</p>
          </div>

          {/* Decay Factor */}
          <div className="space-y-1.5">
            <div className="flex justify-between text-xs">
              <span className="font-medium text-slate-300">Decay Factor</span>
              <span className="font-mono text-amber-400">{decayFactor.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={decayFactor}
              onChange={(e) => setDecayFactor(parseFloat(e.target.value))}
              className="w-full accent-amber-400 bg-slate-950"
            />
            <p className="text-[10px] text-slate-500">Reduces propagated stress across successive hops.</p>
          </div>

          {/* Max Hops */}
          <div className="space-y-1.5">
            <div className="flex justify-between text-xs">
              <span className="font-medium text-slate-300">Maximum Hops</span>
              <span className="font-mono text-slate-200">{maxHops}</span>
            </div>
            <input
              type="range"
              min="1"
              max="5"
              step="1"
              value={maxHops}
              onChange={(e) => setMaxHops(parseInt(e.target.value))}
              className="w-full accent-sky-400 bg-slate-950"
            />
            <p className="text-[10px] text-slate-500">Maximum network distance stress may travel.</p>
          </div>

          {/* Run Button */}
          <button
            onClick={handleManualRun}
            disabled={loading}
            className="w-full bg-sky-500 hover:bg-sky-400 text-slate-950 font-bold py-2.5 px-4 rounded-xl text-xs transition-colors flex items-center justify-center space-x-2 disabled:opacity-50 cursor-pointer"
          >
            <Play className="w-4 h-4 fill-current" />
            <span>{loading ? "Executing Simulation..." : "Run Scenario Simulation"}</span>
          </button>
        </div>

        {/* Right Column: Results Visualization */}
        <div className="lg:col-span-8 space-y-6">
          {loading ? (
            <LoadingState message="Executing controlled frontier stress simulation..." />
          ) : error ? (
            <ErrorState message={error} onRetry={handleManualRun} />
          ) : !simulationResult ? null : (
            <div className="space-y-6">

              {/* Simulation Executive Summary */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <div>
                    <h3 className="text-base font-bold text-slate-100 tracking-tight">
                      Simulation Summary: {simulationResult.source.drug_name}
                    </h3>
                    <p className="text-xs text-slate-400">
                      Source Base Risk: <span className="font-mono font-semibold text-sky-400">{formatScore(simulationResult.source.base_risk_score)}</span>
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-3 text-center">
                  <div className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-800">
                    <div className="text-xs text-slate-400 mb-1">Affected Nodes</div>
                    <div className="text-xl font-bold text-slate-100">
                      {simulationResult.summary.affected_nodes}
                    </div>
                  </div>
                  <div className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-800">
                    <div className="text-xs text-slate-400 mb-1">Mean Downstream Stress</div>
                    <div className="text-xl font-bold text-sky-400 font-mono">
                      {formatScore(simulationResult.summary.mean_downstream_pressure)}
                    </div>
                  </div>
                  <div className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-800">
                    <div className="text-xs text-slate-400 mb-1">Max Downstream Stress</div>
                    <div className="text-xl font-bold text-amber-400 font-mono">
                      {formatScore(simulationResult.summary.max_downstream_pressure)}
                    </div>
                  </div>
                </div>
              </div>

              {/* Affected Nodes Comparison Table */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
                <h3 className="text-base font-bold text-slate-100 tracking-tight">
                  Impacted Drug Cascade ({simulationResult.affected_nodes.length} nodes)
                </h3>

                {simulationResult.affected_nodes.length === 0 ? (
                  <div className="bg-slate-950/60 p-8 rounded-xl border border-slate-800 text-center text-xs text-slate-400 space-y-1">
                    <p className="font-medium text-slate-300">No downstream nodes exceeded the selected pressure threshold.</p>
                    <p className="text-[11px] text-slate-500">This source drug is either isolated or scenario weights/hops are too low.</p>
                  </div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs text-slate-300">
                      <thead className="bg-slate-950/70 border-b border-slate-800 text-slate-400 uppercase tracking-wider font-semibold">
                        <tr>
                          <th className="py-3 px-3">Drug Entity</th>
                          <th className="py-3 px-3">Hop</th>
                          <th className="py-3 px-3">Base Risk</th>
                          <th className="py-3 px-3">Scenario Stress</th>
                          <th className="py-3 px-3">Combined Risk Score</th>
                          <th className="py-3 px-3 w-40">Risk Comparison</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60">
                        {simulationResult.affected_nodes.map((item) => (
                          <tr key={item.node_id} className="hover:bg-slate-800/40 transition-colors">
                            <td className="py-3 px-3 font-semibold text-slate-100">
                              <Link href={`/drugs/${encodeURIComponent(item.node_id)}`} className="hover:text-sky-400 transition-colors">
                                {item.drug_name}
                              </Link>
                              <div className="text-[10px] text-slate-500">{item.therapeutic_category}</div>
                            </td>
                            <td className="py-3 px-3 font-mono text-slate-400">
                              {item.hop_distance !== null && item.hop_distance !== undefined ? `Hop ${item.hop_distance}` : "Source"}
                            </td>
                            <td className="py-3 px-3 font-mono text-slate-400">{formatScore(item.base_risk_score)}</td>
                            <td className="py-3 px-3 font-mono text-sky-400 font-medium">{formatScore(item.scenario_pressure)}</td>
                            <td className="py-3 px-3 font-mono font-bold text-amber-400">{formatScore(item.scenario_risk_score)}</td>
                            <td className="py-3 px-3">
                              {/* Horizontal Visual Risk Bar */}
                              <div className="w-full bg-slate-950 h-2.5 rounded-full overflow-hidden border border-slate-800 flex">
                                <div
                                  className="bg-sky-500 h-full"
                                  style={{ width: `${Math.min(100, item.base_risk_score * 100)}%` }}
                                  title={`Base Risk: ${formatScore(item.base_risk_score)}`}
                                />
                                <div
                                  className="bg-amber-400 h-full"
                                  style={{ width: `${Math.min(100, (item.scenario_risk_score - item.base_risk_score) * 100)}%` }}
                                  title={`Stress Incremental Risk: ${formatScore(item.scenario_risk_score - item.base_risk_score)}`}
                                />
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

              {/* Scientific Boundary Disclaimer */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
                <DisclaimerAlert message={simulationResult.interpretation} />
              </div>

            </div>
          )}
        </div>

      </div>
    </div>
  );
}

export default function ScenarioLabPage() {
  return (
    <Suspense fallback={<LoadingState message="Loading Scenario Lab..." />}>
      <ScenarioLabContent />
    </Suspense>
  );
}
