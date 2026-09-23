import {
  HealthResponse,
  SystemSummary,
  DrugListResponse,
  DrugDetail,
  DrugRelationshipsResponse,
  RiskPredictionResponse,
  ModelInfo,
  GraphSummary,
  ScenarioSimulateRequest,
  ScenarioSimulateResponse,
} from "./types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

async function fetchJson<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  try {
    const res = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
    });

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(
        errorData.detail || `HTTP Error ${res.status}: ${res.statusText}`
      );
    }

    return (await res.json()) as T;
  } catch (err: unknown) {
    if (err instanceof Error) {
      throw err;
    }
    throw new Error("An unexpected network error occurred while connecting to backend.");
  }
}

export const api = {
  getHealth: () => fetchJson<HealthResponse>("/api/health"),
  getSummary: () => fetchJson<SystemSummary>("/api/summary"),
  getDrugs: (params?: {
    search?: string;
    category?: string;
    connected_only?: boolean;
    limit?: number;
    offset?: number;
  }) => {
    const query = new URLSearchParams();
    if (params?.search) query.append("search", params.search);
    if (params?.category) query.append("category", params.category);
    if (params?.connected_only) query.append("connected_only", "true");
    if (params?.limit) query.append("limit", params.limit.toString());
    if (params?.offset) query.append("offset", params.offset.toString());
    const queryString = query.toString() ? `?${query.toString()}` : "";
    return fetchJson<DrugListResponse>(`/api/drugs${queryString}`);
  },
  getDrugDetail: (nodeId: string) =>
    fetchJson<DrugDetail>(`/api/drugs/${encodeURIComponent(nodeId)}`),
  getDrugRisk: (nodeId: string) =>
    fetchJson<RiskPredictionResponse>(`/api/risk/${encodeURIComponent(nodeId)}`),
  getDrugRelationships: (nodeId: string, relationshipType?: string) => {
    const query = relationshipType
      ? `?relationship_type=${encodeURIComponent(relationshipType)}`
      : "";
    return fetchJson<DrugRelationshipsResponse>(
      `/api/drugs/${encodeURIComponent(nodeId)}/relationships${query}`
    );
  },
  getModelInfo: () => fetchJson<ModelInfo>("/api/model/info"),
  getGraphSummary: () => fetchJson<GraphSummary>("/api/graph/summary"),
  simulateScenario: (payload: ScenarioSimulateRequest) =>
    fetchJson<ScenarioSimulateResponse>("/api/scenarios/simulate", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};
