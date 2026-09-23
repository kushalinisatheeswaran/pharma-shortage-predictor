import React from "react";
import { BookOpen, Database, Cpu, Network, ShieldAlert, CheckCircle2 } from "lucide-react";
import { DisclaimerAlert } from "@/components/common/DisclaimerAlert";

export default function MethodologyPage() {
  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* Header Banner */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 md:p-8 space-y-3">
        <div className="inline-flex items-center space-x-2 text-xs font-semibold uppercase tracking-wider text-sky-400 bg-sky-500/10 px-3 py-1 rounded-full border border-sky-500/20">
          <BookOpen className="w-3.5 h-3.5" />
          <span>System Design & Scientific Foundations</span>
        </div>
        <h1 className="text-2xl md:text-3xl font-extrabold text-slate-100 tracking-tight">
          Methodology & Scientific Boundaries
        </h1>
        <p className="text-sm text-slate-300 max-w-3xl leading-relaxed">
          Comprehensive documentation of data sources, entity resolution pipelines, rare-event machine learning evaluation, network graph design, and scenario stress propagation mechanics.
        </p>
      </div>

      {/* Section 1: Data Sources & Boundaries */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
        <div className="flex items-center space-x-3 border-b border-slate-800 pb-3">
          <Database className="w-5 h-5 text-sky-400" />
          <h2 className="text-lg font-bold text-slate-100 tracking-tight">1. Data Sources & Integration</h2>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800 space-y-1.5">
            <div className="font-semibold text-slate-200 text-sm">FDA Shortage Database</div>
            <p className="text-slate-400 leading-relaxed">
              Historical and active shortage records providing event posting dates, resolution statuses, and therapeutic categories.
            </p>
          </div>

          <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800 space-y-1.5">
            <div className="font-semibold text-slate-200 text-sm">CMS Medicare Part D</div>
            <p className="text-slate-400 leading-relaxed">
              Aggregated annual prescription fill volumes, gross drug spending, unique beneficiary counts, and active labeler counts.
            </p>
          </div>

          <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800 space-y-1.5">
            <div className="font-semibold text-slate-200 text-sm">RxNorm Ontology</div>
            <p className="text-slate-400 leading-relaxed">
              Normalized clinical drug nomenclature linking brand/generic entities, active ingredients (IN), and combination concepts (MIN).
            </p>
          </div>
        </div>

        <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800 text-xs text-slate-300 space-y-1">
          <span className="font-semibold text-amber-400">Critical Boundary:</span>
          <p className="text-slate-400 leading-relaxed">
            CMS Part D records represent aggregated annual utilization and spending. They do <strong className="text-slate-200">NOT</strong> provide real-time daily/weekly pharmacy inventory or direct consumer demand.
          </p>
        </div>
      </div>

      {/* Section 2: Entity Resolution & Taxonomy */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
        <div className="flex items-center space-x-3 border-b border-slate-800 pb-3">
          <Network className="w-5 h-5 text-indigo-400" />
          <h2 className="text-lg font-bold text-slate-100 tracking-tight">2. Entity Resolution & Graph Taxonomy</h2>
        </div>

        <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
          <p>
            The project resolves CMS drug names to 484 canonical node entities. Graph B establishes analytical similarity edges across two strict relationship types:
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800 space-y-1.5">
              <div className="font-semibold text-indigo-400">CO_FORMULATED_WITH (311 edges)</div>
              <p className="text-slate-400">
                Represents co-occurrence inside multi-ingredient combination concepts (MIN) in RxNorm (e.g., Amoxicillin and Nystatin in combination concept 1008603).
              </p>
            </div>

            <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800 space-y-1.5">
              <div className="font-semibold text-emerald-400">SAME_THERAPEUTIC_CATEGORY (266 edges)</div>
              <p className="text-slate-400">
                Represents shared membership within specialized FDA therapeutic categories (excluding general unclassified entries).
              </p>
            </div>
          </div>

          <p className="text-slate-400 text-xs bg-slate-950/80 p-3 rounded-xl border border-slate-800">
            <strong className="text-slate-200">Clinical Safety Boundary:</strong> Neither relationship type represents clinical prescribing substitution. Edges explicitly set <code className="text-sky-400">clinical_substitution_supported = FALSE</code>.
          </p>
        </div>
      </div>

      {/* Section 3: Rare-Event Machine Learning */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
        <div className="flex items-center space-x-3 border-b border-slate-800 pb-3">
          <Cpu className="w-5 h-5 text-sky-400" />
          <h2 className="text-lg font-bold text-slate-100 tracking-tight">3. Rare-Event Machine Learning Classifier</h2>
        </div>

        <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
          <p>
            Drug shortage events are rare (~0.79% positive rate in CMS data). The selected model is a balanced Logistic Regression classifier evaluated on temporal validation splits (training 2018–2021, target 2022 shortage event prediction).
          </p>

          <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800 grid grid-cols-2 md:grid-cols-4 gap-3 text-center">
            <div>
              <div className="text-[11px] text-slate-400">PR-AUC</div>
              <div className="text-lg font-bold text-sky-400 font-mono">0.2310</div>
            </div>
            <div>
              <div className="text-[11px] text-slate-400">ROC-AUC</div>
              <div className="text-lg font-bold text-indigo-400 font-mono">0.9169</div>
            </div>
            <div>
              <div className="text-[11px] text-slate-400">Precision (th=0.50)</div>
              <div className="text-lg font-bold text-slate-200 font-mono">0.1628</div>
            </div>
            <div>
              <div className="text-[11px] text-slate-400">Recall (th=0.50)</div>
              <div className="text-lg font-bold text-slate-200 font-mono">0.5385</div>
            </div>
          </div>

          <p className="text-slate-400">
            Because accuracy is uninformative for Imbalanced Datasets, Precision-Recall AUC (PR-AUC) serves as the primary evaluation metric. Model outputs are uncalibrated relative shortage-risk scores.
          </p>
        </div>
      </div>

      {/* Section 4: Controlled Frontier Scenario Stress Propagation */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
        <div className="flex items-center space-x-3 border-b border-slate-800 pb-3">
          <Network className="w-5 h-5 text-amber-400" />
          <h2 className="text-lg font-bold text-slate-100 tracking-tight">4. Controlled Scenario Stress Propagation</h2>
        </div>

        <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
          <p>
            The Scenario Lab models hypothetical inventory pressure cascades using a degree-normalized controlled frontier propagation algorithm:
          </p>

          <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800 font-mono text-[11px] text-sky-300 space-y-1 overflow-x-auto">
            <div>normalized_edge_weight(u, v) = weight / sqrt(degree(u) * degree(v))</div>
            <div>candidate_pressure(v) = pressure(u) * normalized_edge_weight(u, v) * decay_factor</div>
            <div>scenario_risk_score = base_risk_score + scenario_pressure * (1 - base_risk_score)</div>
          </div>

          <ul className="space-y-1.5 list-disc list-inside text-slate-400">
            <li>Frontier-based traversal prevents cyclic feedback loops.</li>
            <li>Zero back-propagation to the source node is strictly enforced.</li>
            <li>All pressure and combined risk scores are clipped to [0, 1].</li>
          </ul>
        </div>
      </div>

      {/* Section 5: System Limitations */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
        <div className="flex items-center space-x-3 border-b border-slate-800 pb-3 text-amber-400">
          <ShieldAlert className="w-5 h-5" />
          <h2 className="text-lg font-bold text-slate-100 tracking-tight">5. System Limitations & Disclaimer</h2>
        </div>

        <div className="space-y-2 text-xs text-slate-400 leading-relaxed">
          <div className="flex items-start space-x-2">
            <CheckCircle2 className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <span><strong className="text-slate-200">Uncalibrated Model Scores:</strong> Output risk scores are relative rankings, not absolute shortage probabilities.</span>
          </div>
          <div className="flex items-start space-x-2">
            <CheckCircle2 className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <span><strong className="text-slate-200">Hypothetical Scenario Analysis:</strong> Stress cascades are user-parameterized scenario explorations, not empirical demand transfers.</span>
          </div>
          <div className="flex items-start space-x-2">
            <CheckCircle2 className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <span><strong className="text-slate-200">No Patient Medical Advice:</strong> This application is research analytical decision-support software. It does NOT recommend medication substitution to patients.</span>
          </div>
        </div>
      </div>

      {/* Global Disclaimer */}
      <DisclaimerAlert />
    </div>
  );
}
