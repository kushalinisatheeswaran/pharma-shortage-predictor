export function formatNumber(val: number | undefined | null): string {
  if (val === undefined || val === null) return "N/A";
  return new Intl.NumberFormat("en-US").format(val);
}

export function formatScore(score: number | undefined | null, decimals = 4): string {
  if (score === undefined || score === null) return "N/A";
  return score.toFixed(decimals);
}

export function formatPercent(val: number | undefined | null, decimals = 2): string {
  if (val === undefined || val === null) return "N/A";
  return `${(val * 100).toFixed(decimals)}%`;
}
