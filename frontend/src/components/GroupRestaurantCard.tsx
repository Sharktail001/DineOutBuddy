import { useState } from "react";
import type { GroupRecommendation } from "../types";
import MenuItems from "./MenuItems";
import s from "../styles/Results.module.css";

const AVATAR_COLORS = [
  "var(--color-accent-1)",
  "var(--color-accent-2)",
  "var(--color-accent-3)",
  "var(--color-accent-4)",
  "var(--color-accent-5)",
];

function scoreClass(score: number): string {
  if (score >= 0.7) return s.scoreGreen;
  if (score >= 0.4) return s.scoreYellow;
  return s.scoreRed;
}

function personScoreColor(score: number): string {
  if (score >= 0.7) return "var(--color-green)";
  if (score >= 0.4) return "var(--color-yellow)";
  return "var(--color-red)";
}

interface Props {
  rec: GroupRecommendation;
  numPeople: number;
}

export default function GroupRestaurantCard({ rec, numPeople }: Props) {
  const [expanded, setExpanded] = useState(false);
  const pct = Math.round(rec.group_score * 100);

  return (
    <div className={s.card} onClick={() => setExpanded(!expanded)}>
      <div className={s.cardTop}>
        <div className={`${s.scoreCircle} ${scoreClass(rec.group_score)}`}>
          {pct}%
        </div>
        <div className={s.cardInfo}>
          <div className={s.cardName}>{rec.name}</div>
          <div className={s.scores}>
            Group Score: {rec.group_score.toFixed(3)}
          </div>
        </div>
      </div>

      <div className={s.personScores}>
        {Array.from({ length: numPeople }, (_, i) => {
          const scores = rec.per_person_scores[String(i)];
          if (!scores) return null;
          const compat = scores.compatibility_score;
          return (
            <div key={i} className={s.personChip}>
              <div
                className={s.personAvatar}
                style={{
                  background: AVATAR_COLORS[i % AVATAR_COLORS.length],
                }}
              >
                P{i + 1}
              </div>
              <span
                className={s.personScore}
                style={{ color: personScoreColor(compat) }}
              >
                {Math.round(compat * 100)}%
              </span>
            </div>
          );
        })}
      </div>

      {expanded && <MenuItems restaurantId={rec.restaurant_id} />}
    </div>
  );
}
