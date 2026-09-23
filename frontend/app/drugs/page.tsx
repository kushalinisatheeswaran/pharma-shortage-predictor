"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Search, Filter, ChevronLeft, ChevronRight, Info, ExternalLink } from "lucide-react";
import { api } from "@/lib/api";
import { DrugSummary } from "@/lib/types";
import { LoadingState } from "@/components/common/LoadingState";
import { ErrorState } from "@/components/common/ErrorState";
import { formatScore, formatNumber } from "@/lib/format";

export default function DrugExplorerPage() {
  const [drugs, setDrugs] = useState<DrugSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState("");
  const [connectedOnly, setConnectedOnly] = useState(false);
  const [page, setPage] = useState(1);
  const limit = 20;

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const categories = [
    "All Categories",
    "Anti-Infective",
    "Cardiovascular",
    "Analgesia/Addiction",
    "Neurology",
    "Endocrinology/Metabolism",
    "Gastroenterology",
    "Oncology",
    "Psychiatry",
    "Rheumatology",
    "Unclassified / General",
  ];

  useEffect(() => {
    let ignore = false;
    const offset = (page - 1) * limit;

    api
      .getDrugs({
        search: search || undefined,
        category: category && category !== "All Categories" ? category : undefined,
        connected_only: connectedOnly,
        limit,
        offset,
      })
      .then((res) => {
        if (!ignore) {
          setDrugs(res.drugs);
          setTotal(res.total);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!ignore) {
          setError(err.message || "Failed to load drugs list.");
          setLoading(false);
        }
      });

    return () => {
      ignore = true;
    };
  }, [search, category, connectedOnly, page]);

  const totalPages = Math.ceil(total / limit) || 1;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight">Drug Explorer</h1>
          <p className="text-xs text-slate-400 mt-1">
            Browse 484 canonical drug nodes with live relative risk scoring and network topology status.
          </p>
        </div>
      </div>

      {/* Filter Bar */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 grid grid-cols-1 md:grid-cols-12 gap-3 items-center">
        {/* Search Field */}
        <div className="md:col-span-5 relative">
          <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search drug name (e.g. AMOXICILLIN)..."
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(1);
            }}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg pl-9 pr-3 py-2 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-sky-500/50"
          />
        </div>

        {/* Category Filter */}
        <div className="md:col-span-4 relative">
          <Filter className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
          <select
            value={category}
            onChange={(e) => {
              setCategory(e.target.value);
              setPage(1);
            }}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg pl-9 pr-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-sky-500/50 appearance-none"
          >
            {categories.map((cat) => (
              <option key={cat} value={cat}>
                {cat}
              </option>
            ))}
          </select>
        </div>

        {/* Connected Only Checkbox */}
        <div className="md:col-span-3 flex items-center space-x-2 pl-1">
          <input
            type="checkbox"
            id="connectedOnly"
            checked={connectedOnly}
            onChange={(e) => {
              setConnectedOnly(e.target.checked);
              setPage(1);
            }}
            className="rounded border-slate-700 bg-slate-950 text-sky-500 focus:ring-sky-500/20"
          />
          <label htmlFor="connectedOnly" className="text-xs text-slate-300 cursor-pointer">
            Connected only (Degree &gt; 0)
          </label>
        </div>
      </div>

      {/* Main Table Content */}
      {loading ? (
        <LoadingState message="Fetching drug catalog from backend..." />
      ) : error ? (
        <ErrorState message={error} onRetry={() => setPage(1)} />
      ) : drugs.length === 0 ? (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-12 text-center text-slate-400 space-y-2">
          <p className="text-sm font-medium">No matching drugs found.</p>
          <p className="text-xs text-slate-500">Try broadening your search term or clearing the category filter.</p>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-950/70 border-b border-slate-800 text-slate-400 uppercase tracking-wider font-semibold">
                  <tr>
                    <th className="py-3.5 px-4">Drug Entity Name</th>
                    <th className="py-3.5 px-4">Therapeutic Category</th>
                    <th className="py-3.5 px-4">
                      <div className="flex items-center space-x-1">
                        <span>Relative Risk Score</span>
                        <span title="Model-generated relative shortage-risk score. This score is not a calibrated probability.">
                          <Info className="w-3.5 h-3.5 text-slate-500" />
                        </span>
                      </div>
                    </th>
                    <th className="py-3.5 px-4">Feature Year</th>
                    <th className="py-3.5 px-4">Graph Relationships</th>
                    <th className="py-3.5 px-4 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {drugs.map((drug) => (
                    <tr key={drug.node_id} className="hover:bg-slate-800/40 transition-colors">
                      <td className="py-3.5 px-4 font-semibold text-slate-100">
                        <Link href={`/drugs/${encodeURIComponent(drug.node_id)}`} className="hover:text-sky-400 transition-colors">
                          {drug.drug_name}
                        </Link>
                        <div className="text-[10px] font-mono text-slate-500">{drug.node_id}</div>
                      </td>
                      <td className="py-3.5 px-4">
                        <span className="bg-slate-800/80 text-slate-300 px-2.5 py-1 rounded text-[11px] border border-slate-700/50">
                          {drug.therapeutic_category}
                        </span>
                      </td>
                      <td className="py-3.5 px-4 font-mono font-medium">
                        <span className={drug.base_risk_score >= 0.5 ? "text-amber-400 font-bold" : "text-sky-400"}>
                          {formatScore(drug.base_risk_score)}
                        </span>
                      </td>
                      <td className="py-3.5 px-4 text-slate-400">{drug.latest_feature_year}</td>
                      <td className="py-3.5 px-4">
                        {drug.is_connected ? (
                          <span className="text-emerald-400 font-medium">
                            {drug.relationship_count} channels
                          </span>
                        ) : (
                          <span className="text-slate-500">Isolated</span>
                        )}
                      </td>
                      <td className="py-3.5 px-4 text-right">
                        <Link
                          href={`/drugs/${encodeURIComponent(drug.node_id)}`}
                          className="inline-flex items-center space-x-1 text-xs text-sky-400 hover:text-sky-300 font-medium"
                        >
                          <span>Inspect</span>
                          <ExternalLink className="w-3.5 h-3.5" />
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Pagination Bar */}
          <div className="flex items-center justify-between bg-slate-900 border border-slate-800 rounded-xl px-4 py-3 text-xs text-slate-400">
            <div>
              Showing <span className="font-semibold text-slate-200">{(page - 1) * limit + 1}</span> to{" "}
              <span className="font-semibold text-slate-200">{Math.min(page * limit, total)}</span> of{" "}
              <span className="font-semibold text-slate-200">{formatNumber(total)}</span> drugs
            </div>
            <div className="flex items-center space-x-2">
              <button
                disabled={page <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                className="p-1.5 rounded-lg border border-slate-800 bg-slate-950 text-slate-300 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-slate-800"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span className="px-2 font-mono text-slate-300">
                Page {page} of {totalPages}
              </span>
              <button
                disabled={page >= totalPages}
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                className="p-1.5 rounded-lg border border-slate-800 bg-slate-950 text-slate-300 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-slate-800"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
