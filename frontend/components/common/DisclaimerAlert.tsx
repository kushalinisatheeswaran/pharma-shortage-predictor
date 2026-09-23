import React from "react";
import { ShieldAlert } from "lucide-react";

interface DisclaimerAlertProps {
  message?: string;
}

export function DisclaimerAlert({
  message = "Hypothetical inventory stress propagation across an analytical drug relationship network. Results do not represent clinical substitution, patient switching, observed demand transfer, causal shortage transmission, or calibrated shortage probabilities.",
}: DisclaimerAlertProps) {
  return (
    <div className="bg-slate-900/90 border border-slate-800/90 rounded-xl p-4 flex items-start space-x-3 text-xs text-slate-400">
      <ShieldAlert className="w-4 h-4 text-sky-400 shrink-0 mt-0.5" />
      <div className="leading-relaxed">
        <span className="font-semibold text-slate-300 mr-1">Scientific Boundary Disclaimer:</span>
        {message}
      </div>
    </div>
  );
}
