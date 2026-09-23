"use client";

import { useEffect, useState, use } from "react";
import Link from "next/link";
import { 
  ArrowLeft, 
  Pill, 
  Layers, 
  Cpu, 
  Building2, 
  History, 
  ShieldAlert, 
  ExternalLink,
  Info
} from "lucide-react";
import { api } from "@/lib/api";
import { DrugDetail, RiskPredictionResponse, DrugRelationshipsResponse } from "@/lib/types";
import { LoadingState } from "@/components/common/LoadingState";
import { ErrorState } from "@/components/common/ErrorState";
import { formatScore } from "@/lib/format";

export default function DrugDetailPage({ params }: { params: Promise<{ nodeId: string }> }) {
  const resolvedParams = use(params);
  const nodeId = decodeURIComponent(resolvedParams.nodeId);

  const [detail, setDetail] = useState<DrugDetail | null>(null);
  const [risk, setRisk] = useState<RiskPredictionResponse | null>(null);
  const [relationships, setRelationships] = useState<DrugRelationshipsResponse | null>(null);
  const [relFilter, setRelFilter] = useState<string>("ALL");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let ignore = false;
    Promise.all([
      api.getDrugDetail(nodeId),
      api.getDrugRisk(nodeId),
      api.getDrugRelationships(nodeId),
    ])
      .then(([detailRes, riskRes, relRes]) => {
        if (!ignore) {
          setDetail(detailRes);
          setRisk(riskRes);
          setRelationships(relRes);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!ignore) {
          setError(err.message || `Failed to load details for drug node '${nodeId}'.`);
          setLoading(false);
        }
      });

    return () => {
      ignore = true;
    };
  }, [nodeId]);

  if (loading) return <LoadingState message={`Loading intelligence for ${nodeId}...`} />;
  if (error) return <ErrorState message={error} onRetry={() => window.location.reload()} />;
  if (!detail || !risk || !relationships) return null;

  const filteredRelationships = relationships.relationships.filter((item) => {
    if (relFilter === "ALL") return true;
    return item.relationship_type === relFilter;
  });

  return (
    <div className="space-y-6">
      {/* Navigation & Back Button */}
      <div>
        <Link
          href="/drugs"
          className="inline-flex items-center space-x-2 text-xs font-medium text-slate-400 hover:text-slate-200 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to Drug Explorer</span>
        </Link>
      </div>

      {/* Header Banner */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 md:p-8 space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center space-x-3">
              <h1 className="text-2xl md:text-3xl font-extrabold text-slate-100 tracking-tight">
                {detail.drug_name}
              </h1>
              <span className="bg-slate-800 text-slate-300 text-xs px-3 py-1 rounded-full border border-slate-700 font-mono">
                {detail.node_id}
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Therapeutic Category: <span className="text-slate-200 font-medium">{detail.therapeutic_category}</span>
            </p>
          </div>

          <div className="flex items-center space-x-3">
            <Link
              href={`/scenarios?source=${encodeURIComponent(detail.node_id)}`}
              className="inline-flex items-center space-x-2 bg-sky-500/10 hover:bg-sky-500/20 text-sky-400 border border-sky-500/20 text-xs font-semibold px-4 py-2 rounded-xl transition-colors"
            >
              <Layers className="w-4 h-4" />
              <span>Simulate Stress Cascade</span>
            </Link>
          </div>
        </div>

        {/* Metadata Chips */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
          <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
            <div className="text-[11px] text-slate-400 flex items-center space-x-1">
              <Pill className="w-3.5 h-3.5 text-sky-400" />
              <span>Canonical RxCUI</span>
            </div>
            <div className="text-sm font-mono font-bold text-slate-200 mt-1">{detail.canonical_rxcui}</div>
          </div>
          <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
            <div className="text-[11px] text-slate-400 flex items-center space-x-1">
              <Building2 className="w-3.5 h-3.5 text-indigo-400" />
              <span>Active Manufacturers</span>
            </div>
            <div className="text-sm font-bold text-slate-200 mt-1">
              {detail.manufacturer_count}{" "}
              {detail.single_manufacturer_flag === 1 && (
                <span className="text-[10px] text-amber-400 font-normal">(Sole Reliance)</span>
              )}
            </div>
          </div>
          <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
            <div className="text-[11px] text-slate-400 flex items-center space-x-1">
              <History className="w-3.5 h-3.5 text-amber-400" />
              <span>Historical Shortages</span>
            </div>
            <div className="text-sm font-bold text-slate-200 mt-1">{detail.historical_shortage_count} events</div>
          </div>
          <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
            <div className="text-[11px] text-slate-400 flex items-center space-x-1">
              <Layers className="w-3.5 h-3.5 text-emerald-400" />
              <span>Graph Degree</span>
            </div>
            <div className="text-sm font-bold text-slate-200 mt-1">{detail.relationship_count} channels</div>
          </div>
        </div>
      </div>

      {/* Main Grid: Live ML Model Panel + Relationships Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* Live Packaged ML Inference Panel */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
          <div className="flex items-center space-x-3 border-b border-slate-800 pb-4">
            <div className="p-2 bg-sky-500/10 text-sky-400 rounded-xl border border-sky-500/20">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-slate-100 tracking-tight">
                Live Packaged ML Inference
              </h2>
              <p className="text-xs text-slate-400">Scikit-Learn Pipeline Artifact Execution</p>
            </div>
          </div>

          <div className="bg-slate-950/80 p-5 rounded-xl border border-slate-800 flex items-center justify-between">
            <div>
              <div className="text-xs text-slate-400 uppercase tracking-wider font-medium">
                Uncalibrated Relative Risk Score
              </div>
              <div className="text-3xl font-extrabold text-sky-400 tracking-tight mt-1 font-mono">
                {formatScore(risk.base_risk_score)}
              </div>
              <div className="text-[11px] text-slate-500 mt-1">
                Feature Observation Year: <span className="text-slate-300 font-medium">{risk.feature_year}</span>
              </div>
            </div>

            <div className="text-right">
              <span className="inline-block bg-sky-500/10 text-sky-400 text-xs px-3 py-1 rounded-full border border-sky-500/20 font-medium">
                Live Model Output
              </span>
            </div>
          </div>

          <div className="space-y-2 text-xs">
            <div className="flex justify-between py-1.5 border-b border-slate-800/60">
              <span className="text-slate-400">Packaged Model Name:</span>
              <span className="font-semibold text-slate-200">{risk.model_name}</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-800/60">
              <span className="text-slate-400">Score Consistency Check:</span>
              <span className="text-emerald-400 font-medium">{risk.consistency_status}</span>
            </div>
          </div>

          <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800 text-xs text-slate-400 space-y-2">
            <div className="flex items-center space-x-2 text-slate-300 font-semibold">
              <Info className="w-4 h-4 text-sky-400" />
              <span>Scientific & Model Boundaries</span>
            </div>
            <p className="leading-relaxed text-[11px]">
              {risk.score_interpretation}
            </p>
            <p className="text-[10px] text-slate-500">
              Generated using the packaged MedCascade Logistic Regression inference pipeline.
            </p>
          </div>
        </div>

        {/* Local Network Topology View */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div>
              <h2 className="text-lg font-bold text-slate-100 tracking-tight">
                Graph B Local Topology
              </h2>
              <p className="text-xs text-slate-400">
                Connected Neighbors ({detail.relationship_count} channels)
              </p>
            </div>
            
            {/* Relationship Filter */}
            <select
              value={relFilter}
              onChange={(e) => setRelFilter(e.target.value)}
              className="bg-slate-950 border border-slate-800 text-xs text-slate-300 px-3 py-1.5 rounded-lg focus:outline-none"
            >
              <option value="ALL">All Relationships</option>
              <option value="CO_FORMULATED_WITH">Co-formulated Only</option>
              <option value="SAME_THERAPEUTIC_CATEGORY">Category Only</option>
            </select>
          </div>

          {filteredRelationships.length === 0 ? (
            <div className="bg-slate-950/60 p-8 rounded-xl border border-slate-800 text-center text-xs text-slate-400 space-y-1">
              <p className="font-medium">No relationship channels match the selected filter.</p>
              <p className="text-[11px] text-slate-500">
                {detail.relationship_count === 0 ? "This drug node is isolated in Graph B (degree = 0)." : "Try selecting 'All Relationships'."}
              </p>
            </div>
          ) : (
            <div className="space-y-2 max-h-[340px] overflow-y-auto pr-1">
              {filteredRelationships.map((item, idx) => (
                <div
                  key={`${item.related_node_id}-${item.relationship_type}-${idx}`}
                  className="bg-slate-950/60 border border-slate-800/80 hover:border-slate-700/80 rounded-xl p-3.5 flex items-center justify-between transition-colors"
                >
                  <div className="space-y-0.5">
                    <Link
                      href={`/drugs/${encodeURIComponent(item.related_node_id)}`}
                      className="font-semibold text-xs text-slate-100 hover:text-sky-400 transition-colors flex items-center space-x-1.5"
                    >
                      <span>{item.drug_name}</span>
                      <ExternalLink className="w-3 h-3 text-slate-500" />
                    </Link>
                    <div className="text-[10px] font-mono text-slate-500">{item.related_node_id}</div>
                  </div>

                  <div className="flex items-center space-x-3 text-right">
                    <div>
                      <span
                        className={`text-[10px] font-medium px-2 py-0.5 rounded border ${
                          item.relationship_type === "CO_FORMULATED_WITH"
                            ? "bg-indigo-500/10 text-indigo-400 border-indigo-500/20"
                            : "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                        }`}
                      >
                        {item.relationship_type === "CO_FORMULATED_WITH" ? "Co-formulated" : "Same Category"}
                      </span>
                    </div>
                    <div className="text-xs font-mono font-semibold text-sky-400">
                      {formatScore(item.base_risk_score)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          <div className="bg-slate-950/80 p-3 rounded-xl border border-slate-800 text-[11px] text-slate-400 flex items-start space-x-2">
            <ShieldAlert className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
            <p>
              Relationship channels preserve structural co-occurrence and category similarity. Multi-edges for identical node pairs are preserved.
            </p>
          </div>
        </div>

      </div>
    </div>
  );
}
