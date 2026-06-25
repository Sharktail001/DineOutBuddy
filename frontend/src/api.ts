import type {
  TasteProfile,
  Recommendation,
  GroupRecommendation,
  PaginatedMenu,
} from "./types";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json();
}

export function fetchRecommendations(
  profile: TasteProfile,
  limit: number
): Promise<Recommendation[]> {
  return request("/recommend", {
    method: "POST",
    body: JSON.stringify({ ...profile, limit }),
  });
}

export function fetchGroupRecommendations(
  profiles: TasteProfile[],
  limit: number
): Promise<GroupRecommendation[]> {
  return request("/recommend/group", {
    method: "POST",
    body: JSON.stringify({ profiles, limit }),
  });
}

export function fetchMenu(
  restaurantId: string,
  page: number,
  pageSize: number
): Promise<PaginatedMenu> {
  return request(
    `/restaurants/${restaurantId}/menu?page=${page}&page_size=${pageSize}`
  );
}
