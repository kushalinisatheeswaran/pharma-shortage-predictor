"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { 
  Pill, 
  Network, 
  Layers, 
  Target, 
  ArrowRight, 
  BarChart3, 
  Info,
  ShieldAlert
} from "lucide-react";
import { api } from "@/lib/api";
import { SystemSummary, ModelInfo, GraphSummary } from "@/lib/types";
import { StatCard } from "@/components/common/StatCard";
import { LoadingState } from "@/components/common/LoadingState";
import { ErrorState } from "@/components/common/ErrorState";
import { DisclaimerAlert } from "@/components/common/DisclaimerAlert";
import { formatNumber, formatScore, formatPercent } from "@/lib/format";

export default function OverviewPage() {
  const [summary, setSummary] = useState<SystemSummary | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [graphSummary, setGraphSummary] = useState<GraphSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let ignore = false;
    Promise.all([api.getSummary(), api.getModelInfo(), api.getGraphSummary()])
      .then(([sumRes, modelRes, graphRes]) => {
        if (!ignore) {
          setSummary(sumRes);
          setModelInfo(modelRes);
          setGraphSummary(graphRes);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!ignore) {
          setError(err.message || "Failed to load dashboard overview data.");
          setLoading(false);
        }
      });

    return () => {
      ignore = true;
    };
  }, []);

  if (loading) return <LoadingState message="Loading MedCascade system overview..." />;
  if (error) return <ErrorState message={error} onRetry={() => window.location.reload()} />;
  if (!summary || !modelInfo || !graphSummary) return null;

  const vm = modelInfo.validation_metrics;

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 md:p-8 space-y-3">
        <div className="inline-flex items-center space-x-2 text-xs font-semibold uppercase tracking-wider text-sky-400 bg-sky-500/10 px-3 py-1 rounded-full border border-sky-500/20">
          <Target className="w-3.5 h-3.5" />
          <span>Analytical Intelligence Platform</span>
        </div>
        <h1 className="text-2xl md:text-3xl font-extrabold text-slate-100 tracking-tight">
          MedCascade Overview
        </h1>
        <p className="text-sm text-slate-300 max-w-3xl leading-relaxed">
          An experimental analytics platform combining Medicare Part D utilization, FDA shortage observations, machine-learning relative risk scoring, and scenario-based network stress analysis.
        </p>
      </div>

      {/* KPI Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Tracked Canonical Drugs"
          value={formatNumber(summary.total_drugs)}
          subtitle={`${summary.mapped_drugs} linked to RxNorm RxCUIs`}
          icon={Pill}
        />
        <StatCard
          title="Connected Drug Nodes"
          value={formatNumber(summary.connected_nodes)}
          subtitle={`${summary.isolated_nodes} isolated nodes (0 edges)`}
          icon={Network}
        />
        <StatCard
          title="Relationship Channels"
          value={formatNumber(summary.relationship_channels)}
          subtitle={`${summary.unique_node_pairs} unique drug pairs`}
          icon={Layers}
        />
        <StatCard
          title="Model Validation PR-AUC"
          value={formatScore(vm.pr_auc)}
          subtitle={`ROC-AUC ${formatScore(vm.roc_auc)} | Recall ${formatScore(vm.recall)}`}
          icon={BarChart3}
        />
      </div>

      {/* Main Grid: Model Performance + Network Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* ML Model Performance Panel */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div>
              <h2 className="text-lg font-bold text-slate-100 tracking-tight">
                ML Shortage Risk Model
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">{modelInfo.model_name}</p>
            </div>
            <span className="text-[11px] font-mono bg-slate-800 text-slate-300 px-2.5 py-1 rounded border border-slate-700">
              {modelInfo.model_version}
            </span>
          </div>

          <div className="grid grid-cols-3 gap-3 text-center">
            <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800">
              <div className="text-xs text-slate-400 mb-1">PR-AUC</div>
              <div className="text-lg font-bold text-sky-400">{formatScore(vm.pr_auc)}</div>
            </div>
            <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800">
              <div className="text-xs text-slate-400 mb-1">ROC-AUC</div>
              <div className="text-lg font-bold text-indigo-400">{formatScore(vm.roc_auc)}</div>
            </div>
            <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800">
              <div className="text-xs text-slate-400 mb-1">F1 Score</div>
              <div className="text-lg font-bold text-emerald-400">{formatScore(vm.f1)}</div>
            </div>
          </div>

          <div className="space-y-2 text-xs">
            <div className="flex justify-between py-1.5 border-b border-slate-800/60">
              <span className="text-slate-400">Precision (at th=0.50):</span>
              <span className="font-semibold text-slate-200">{formatScore(vm.precision)}</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-800/60">
              <span className="text-slate-400">Recall (at th=0.50):</span>
              <span className="font-semibold text-slate-200">{formatScore(vm.recall)}</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-800/60">
              <span className="text-slate-400">Temporal Validation Outcomes:</span>
              <span className="font-mono text-slate-200">TP={vm.tp} | FP={vm.fp} | FN={vm.fn}</span>
            </div>
            <div className="flex justify-between py-1.5">
              <span className="text-slate-400">Rare-Event Context:</span>
              <span className="text-slate-300">
                {modelInfo.training_positive_count} positives out of {formatNumber(modelInfo.training_observations)} obs ({formatPercent(modelInfo.training_positive_count / modelInfo.training_observations)})
              </span>
            </div>
          </div>

          <div className="bg-slate-950/80 p-3.5 rounded-xl border border-slate-800 text-xs text-slate-400 flex items-start space-x-2.5">
            <Info className="w-4 h-4 text-sky-400 shrink-0 mt-0.5" />
            <p className="leading-relaxed">
              Model output is an <strong className="text-slate-200 font-semibold">uncalibrated relative shortage-risk score</strong>. It represents relative vulnerability ranking, not an absolute shortage probability.
            </p>
          </div>
        </div>

        {/* Graph B Network Overview Panel */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div>
              <h2 className="text-lg font-bold text-slate-100 tracking-tight">
                Graph B Network Architecture
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">Analytical Similarity Topology</p>
            </div>
            <Link
              href="/scenarios"
              className="inline-flex items-center space-x-1.5 text-xs font-semibold text-sky-400 hover:text-sky-300 bg-sky-500/10 hover:bg-sky-500/20 px-3 py-1.5 rounded-lg transition-colors border border-sky-500/20"
            >
              <span>Launch Lab</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-800">
              <div className="text-xs text-slate-400">Co-formulated Edges</div>
              <div className="text-xl font-bold text-slate-100 mt-1">
                {formatNumber(graphSummary.coformulated_relationships)}
              </div>
              <p className="text-[11px] text-slate-500 mt-0.5">Multi-ingredient concept co-occurrence</p>
            </div>
            <div className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-800">
              <div className="text-xs text-slate-400">Therapeutic Category Edges</div>
              <div className="text-xl font-bold text-slate-100 mt-1">
                {formatNumber(graphSummary.therapeutic_category_relationships)}
              </div>
              <p className="text-[11px] text-slate-500 mt-0.5">Domain category co-membership</p>
            </div>
          </div>

          <div className="space-y-2 text-xs">
            <div className="flex justify-between py-1.5 border-b border-slate-800/60">
              <span className="text-slate-400">Total Graph B Nodes:</span>
              <span className="font-semibold text-slate-200">{formatNumber(graphSummary.nodes)}</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-800/60">
              <span className="text-slate-400">Unique Node Pairs:</span>
              <span className="font-semibold text-slate-200">{formatNumber(graphSummary.unique_node_pairs)}</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-800/60">
              <span className="text-slate-400">Connected Components:</span>
              <span className="font-semibold text-slate-200">{formatNumber(graphSummary.connected_components)}</span>
            </div>
            <div className="flex justify-between py-1.5">
              <span className="text-slate-400">Largest Component Size:</span>
              <span className="font-semibold text-sky-400">{formatNumber(graphSummary.largest_component_size)} nodes</span>
            </div>
          </div>

          <div className="bg-slate-950/80 p-3.5 rounded-xl border border-slate-800 text-xs text-slate-400 flex items-start space-x-2.5">
            <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <p className="leading-relaxed">
              Graph B relationships represent structural analytical co-occurrence and category similarity. They do <strong className="text-slate-200 font-semibold">NOT</strong> represent clinical prescribing substitution.
            </p>
          </div>
        </div>

      </div>

      {/* Global Disclaimer Alert */}
      <DisclaimerAlert />
    </div>
  );
}
