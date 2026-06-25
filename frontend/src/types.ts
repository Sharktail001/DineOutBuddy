export interface TasteProfile {
  cuisine_preferences: string[];
  spice_tolerance: number;
  price_preference: number;
  dietary_requirements: string[];
  dietary_preferences: string[];
}

export interface Recommendation {
  restaurant_id: string;
  name: string;
  address: string | null;
  cuisine_tags: string[];
  final_score: number;
  similarity_score: number;
  compatibility_score: number;
}

export interface PersonScores {
  final_score: number;
  compatibility_score: number;
  similarity_score: number;
}

export interface GroupRecommendation {
  restaurant_id: string;
  name: string;
  group_score: number;
  per_person_scores: Record<string, PersonScores>;
}

export interface MenuItem {
  id: string;
  category: string | null;
  name: string;
  description: string | null;
  price: number | null;
}

export interface PaginatedMenu {
  items: MenuItem[];
  page: number;
  page_size: number;
  total: number;
}

export const CUISINE_OPTIONS = [
  "american",
  "mediterranean",
  "indian",
  "mexican",
  "italian",
  "chinese",
  "korean",
  "japanese",
  "thai",
] as const;

export const DIETARY_OPTIONS = [
  "halal",
  "vegan",
  "kosher",
  "vegetarian",
  "gluten_free",
] as const;

export const DIETARY_LABELS: Record<string, string> = {
  halal: "Halal",
  vegan: "Vegan",
  kosher: "Kosher",
  vegetarian: "Vegetarian",
  gluten_free: "Gluten Free",
};

export function emptyProfile(): TasteProfile {
  return {
    cuisine_preferences: [],
    spice_tolerance: 3,
    price_preference: 2,
    dietary_requirements: [],
    dietary_preferences: [],
  };
}
