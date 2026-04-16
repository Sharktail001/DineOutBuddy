CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS restaurants (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    address TEXT,
    lat DOUBLE PRECISION,
    lng DOUBLE PRECISION,
    city TEXT,
    zip TEXT,
    price_level INT,
    google_place_id TEXT,
    yelp_id TEXT,
    yelp_url TEXT,
    ubereats_id TEXT,
    website TEXT,
    phone TEXT,
    hours TEXT,
    rating_score DOUBLE PRECISION,
    rating_count INT,
    data_quality_score DOUBLE PRECISION DEFAULT 0,
    last_verified_at TIMESTAMP,
    needs_refresh BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS menu_items (
    id UUID PRIMARY KEY,
    restaurant_id UUID NOT NULL REFERENCES restaurants(id) ON DELETE CASCADE,
    category TEXT,
    name TEXT NOT NULL,
    description TEXT,
    price DOUBLE PRECISION,
    source TEXT,
    fetched_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dietary_attributes (
    id UUID PRIMARY KEY,
    restaurant_id UUID NOT NULL REFERENCES restaurants(id) ON DELETE CASCADE,
    dietary_type TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence_tier INT,
    source TEXT,
    source_url TEXT,
    notes TEXT,
    fetched_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cuisine_tags (
    restaurant_id UUID NOT NULL REFERENCES restaurants(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    source TEXT,
    derived_from_menu BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (restaurant_id, tag, source)
);

CREATE TABLE IF NOT EXISTS feature_vectors (
    restaurant_id UUID PRIMARY KEY REFERENCES restaurants(id) ON DELETE CASCADE,
    vector VECTOR,
    vector_version TEXT,
    computed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY,
    name TEXT,
    email TEXT UNIQUE,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS taste_profiles (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    spice_tolerance INT,
    price_preference INT,
    ambiance_preference TEXT[],
    cuisine_preferences TEXT[],
    cuisine_dislikes TEXT[],
    dietary_requirements TEXT[],
    dietary_preferences TEXT[],
    adventure_score INT,
    profile_vector VECTOR,
    last_updated TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_interactions (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    restaurant_id UUID NOT NULL REFERENCES restaurants(id) ON DELETE CASCADE,
    menu_item_id UUID REFERENCES menu_items(id) ON DELETE SET NULL,
    interaction_type TEXT NOT NULL,
    rating INT,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS group_sessions (
    id UUID PRIMARY KEY,
    name TEXT,
    user_ids UUID[],
    location_lat DOUBLE PRECISION,
    location_lng DOUBLE PRECISION,
    radius_miles INT,
    status TEXT,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS group_recommendations (
    id UUID PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES group_sessions(id) ON DELETE CASCADE,
    restaurant_id UUID NOT NULL REFERENCES restaurants(id) ON DELETE CASCADE,
    group_score DOUBLE PRECISION,
    per_user_scores JSONB,
    per_user_items JSONB,
    rank INT,
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS data_fetch_log (
    id UUID PRIMARY KEY,
    source TEXT NOT NULL,
    query TEXT,
    status TEXT,
    fetched_at TIMESTAMP,
    records_added INT DEFAULT 0,
    records_updated INT DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_restaurants_google_place_id
    ON restaurants (google_place_id);

CREATE INDEX IF NOT EXISTS idx_restaurants_yelp_id
    ON restaurants (yelp_id);

CREATE INDEX IF NOT EXISTS idx_restaurants_ubereats_id
    ON restaurants (ubereats_id);

CREATE INDEX IF NOT EXISTS idx_menu_items_restaurant_id
    ON menu_items (restaurant_id);

CREATE INDEX IF NOT EXISTS idx_dietary_attributes_restaurant_id
    ON dietary_attributes (restaurant_id);

CREATE INDEX IF NOT EXISTS idx_cuisine_tags_restaurant_id
    ON cuisine_tags (restaurant_id);

CREATE INDEX IF NOT EXISTS idx_taste_profiles_user_id
    ON taste_profiles (user_id);

CREATE INDEX IF NOT EXISTS idx_user_interactions_user_id
    ON user_interactions (user_id);

CREATE INDEX IF NOT EXISTS idx_user_interactions_restaurant_id
    ON user_interactions (restaurant_id);

CREATE INDEX IF NOT EXISTS idx_group_recommendations_session_id
    ON group_recommendations (session_id);

CREATE INDEX IF NOT EXISTS idx_data_fetch_log_source_fetched_at
    ON data_fetch_log (source, fetched_at);
