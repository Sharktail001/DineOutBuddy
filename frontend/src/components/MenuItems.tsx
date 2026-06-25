import { useState, useEffect } from "react";
import { fetchMenu } from "../api";
import type { MenuItem } from "../types";
import s from "../styles/Results.module.css";

interface Props {
  restaurantId: string;
}

export default function MenuItems({ restaurantId }: Props) {
  const [items, setItems] = useState<MenuItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchMenu(restaurantId, 1, 20)
      .then((data) => {
        if (!cancelled) {
          setItems(data.items);
          setTotal(data.total);
        }
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [restaurantId]);

  if (loading) return <div className={s.menuLoading}>Loading menu...</div>;
  if (error) return <div className={s.menuLoading}>Failed to load menu</div>;
  if (items.length === 0)
    return <div className={s.menuLoading}>No menu items available</div>;

  return (
    <div className={s.menuSection}>
      <div className={s.menuTitle}>Menu Items</div>
      {items.map((item) => (
        <div key={item.id} className={s.menuItem}>
          <div>
            <div className={s.menuItemName}>{item.name}</div>
            {item.description && (
              <div className={s.menuItemDesc}>{item.description}</div>
            )}
          </div>
          {item.price != null && (
            <span className={s.menuItemPrice}>${item.price.toFixed(2)}</span>
          )}
        </div>
      ))}
      {total > items.length && (
        <div className={s.menuMore}>
          Showing {items.length} of {total} items
        </div>
      )}
    </div>
  );
}
