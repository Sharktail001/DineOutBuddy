import { useCallback } from "react";
import type { TasteProfile } from "../types";
import {
  CUISINE_OPTIONS,
  DIETARY_OPTIONS,
  DIETARY_LABELS,
} from "../types";
import s from "../styles/PreferenceForm.module.css";

const PRICE_LABELS = ["$", "$$", "$$$", "$$$$"];

const AVATAR_COLORS = [
  "var(--color-accent-1)",
  "var(--color-accent-2)",
  "var(--color-accent-3)",
  "var(--color-accent-4)",
  "var(--color-accent-5)",
];

interface Props {
  profile: TasteProfile;
  onChange: (p: TasteProfile) => void;
  personIndex?: number;
  onRemove?: () => void;
}

export default function PreferenceForm({
  profile,
  onChange,
  personIndex,
  onRemove,
}: Props) {
  const toggleCuisine = useCallback(
    (tag: string) => {
      const prefs = profile.cuisine_preferences.includes(tag)
        ? profile.cuisine_preferences.filter((t) => t !== tag)
        : [...profile.cuisine_preferences, tag];
      onChange({ ...profile, cuisine_preferences: prefs });
    },
    [profile, onChange]
  );

  const toggleDietary = useCallback(
    (dtype: string) => {
      const reqs = profile.dietary_requirements.includes(dtype)
        ? profile.dietary_requirements.filter((d) => d !== dtype)
        : [...profile.dietary_requirements, dtype];
      onChange({ ...profile, dietary_requirements: reqs });
    },
    [profile, onChange]
  );

  const isGroup = personIndex !== undefined;
  const color = isGroup
    ? AVATAR_COLORS[personIndex % AVATAR_COLORS.length]
    : undefined;

  return (
    <div className={s.form}>
      {isGroup && (
        <div className={s.personHeader}>
          <div className={s.personBadge}>
            <div className={s.avatar} style={{ background: color }}>
              P{personIndex + 1}
            </div>
            <span className={s.personLabel}>Person {personIndex + 1}</span>
          </div>
          {onRemove && (
            <button className={s.removeBtn} onClick={onRemove}>
              Remove
            </button>
          )}
        </div>
      )}

      <div className={s.section}>
        <span className={s.label}>Cuisines</span>
        <div className={s.cuisineGrid}>
          {CUISINE_OPTIONS.map((c) => (
            <button
              key={c}
              type="button"
              className={`${s.cuisineChip} ${
                profile.cuisine_preferences.includes(c)
                  ? s.cuisineChipActive
                  : ""
              }`}
              onClick={() => toggleCuisine(c)}
            >
              {c.replace(/_/g, " ")}
            </button>
          ))}
        </div>
      </div>

      <div className={s.section}>
        <span className={s.label}>Spice Tolerance</span>
        <div className={s.sliderRow}>
          <span className={s.sliderValue}>1</span>
          <input
            type="range"
            min={1}
            max={5}
            value={profile.spice_tolerance}
            className={s.slider}
            onChange={(e) =>
              onChange({ ...profile, spice_tolerance: Number(e.target.value) })
            }
          />
          <span className={s.sliderValue}>{profile.spice_tolerance}</span>
        </div>
      </div>

      <div className={s.section}>
        <span className={s.label}>Price Preference</span>
        <div className={s.priceRow}>
          {PRICE_LABELS.map((label, i) => (
            <button
              key={i}
              type="button"
              className={`${s.priceBtn} ${
                profile.price_preference === i + 1 ? s.priceBtnActive : ""
              }`}
              onClick={() => onChange({ ...profile, price_preference: i + 1 })}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className={s.section}>
        <span className={s.label}>Dietary Requirements</span>
        <div className={s.dietaryGrid}>
          {DIETARY_OPTIONS.map((d) => (
            <button
              key={d}
              type="button"
              className={`${s.dietaryChip} ${
                profile.dietary_requirements.includes(d)
                  ? s.dietaryChipActive
                  : ""
              }`}
              onClick={() => toggleDietary(d)}
            >
              {DIETARY_LABELS[d]}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
