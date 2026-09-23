import React from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";

interface ErrorStateProps {
  title?: string;
  message: string;
  onRetry?: () => void;
}

export function ErrorState({ title = "Backend Connection Error", message, onRetry }: ErrorStateProps) {
  return (
    <div className="bg-amber-950/20 border border-amber-500/30 rounded-xl p-6 text-slate-200 my-4 space-y-3">
      <div className="flex items-center space-x-3 text-amber-400">
        <AlertTriangle className="w-5 h-5 shrink-0" />
        <h3 className="font-semibold text-sm tracking-tight">{title}</h3>
      </div>
      <p className="text-xs text-slate-300 leading-relaxed">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="inline-flex items-center space-x-2 text-xs font-medium text-amber-300 hover:text-amber-100 bg-amber-500/20 hover:bg-amber-500/30 px-3 py-1.5 rounded-lg transition-colors border border-amber-500/30"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Retry Connection</span>
        </button>
      )}
    </div>
  );
}
