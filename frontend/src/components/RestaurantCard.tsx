import { useState } from "react";
import type { Recommendation } from "../types";
import MenuItems from "./MenuItems";
import s from "../styles/Results.module.css";

function scoreClass(score: number): string {
  if (score >= 0.7) return s.scoreGreen;
  if (score >= 0.4) return s.scoreYellow;
  return s.scoreRed;
}

interface Props {
  rec: Recommendation;
}

export default function RestaurantCard({ rec }: Props) {
  const [expanded, setExpanded] = useState(false);
  const pct = Math.round(rec.compatibility_score * 100);

  return (
    <div className={s.card} onClick={() => setExpanded(!expanded)}>
      <div className={s.cardTop}>
        <div className={`${s.scoreCircle} ${scoreClass(rec.compatibility_score)}`}>
          {pct}%
        </div>
        <div className={s.cardInfo}>
          <div className={s.cardName}>{rec.name}</div>
          {rec.address && <div className={s.cardAddress}>{rec.address}</div>}
          <div className={s.tags}>
            {rec.cuisine_tags.slice(0, 6).map((t) => (
              <span key={t} className={s.tag}>
                {t.replace(/_/g, " ")}
              </span>
            ))}
          </div>
          <div className={s.scores}>
            Score: {rec.final_score.toFixed(3)} &middot; Similarity:{" "}
            {rec.similarity_score.toFixed(3)}
          </div>
        </div>
      </div>
      {expanded && <MenuItems restaurantId={rec.restaurant_id} />}
    </div>
  );
}
