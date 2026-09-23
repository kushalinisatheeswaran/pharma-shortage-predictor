import React from "react";

interface LoadingStateProps {
  message?: string;
}

export function LoadingState({ message = "Loading backend intelligence..." }: LoadingStateProps) {
  return (
    <div className="py-16 flex flex-col items-center justify-center space-y-4">
      <div className="w-8 h-8 border-3 border-sky-400/30 border-t-sky-400 rounded-full animate-spin" />
      <p className="text-sm text-slate-400 font-medium">{message}</p>
    </div>
  );
}
