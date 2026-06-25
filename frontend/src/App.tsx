import { useState, useCallback } from "react";
import type {
  TasteProfile,
  Recommendation,
  GroupRecommendation,
} from "./types";
import { emptyProfile } from "./types";
import { fetchRecommendations, fetchGroupRecommendations } from "./api";
import PreferenceForm from "./components/PreferenceForm";
import RestaurantCard from "./components/RestaurantCard";
import GroupRestaurantCard from "./components/GroupRestaurantCard";
import s from "./styles/App.module.css";
import r from "./styles/Results.module.css";

type Mode = "solo" | "group";

export default function App() {
  const [mode, setMode] = useState<Mode>("solo");

  const [soloProfile, setSoloProfile] = useState<TasteProfile>(emptyProfile());
  const [soloResults, setSoloResults] = useState<Recommendation[]>([]);
  const [soloLoading, setSoloLoading] = useState(false);
  const [soloError, setSoloError] = useState<string | null>(null);
  const [soloSearched, setSoloSearched] = useState(false);

  const [groupProfiles, setGroupProfiles] = useState<TasteProfile[]>([
    emptyProfile(),
    emptyProfile(),
  ]);
  const [groupResults, setGroupResults] = useState<GroupRecommendation[]>([]);
  const [groupLoading, setGroupLoading] = useState(false);
  const [groupError, setGroupError] = useState<string | null>(null);
  const [groupSearched, setGroupSearched] = useState(false);

  const updateGroupProfile = useCallback((index: number, p: TasteProfile) => {
    setGroupProfiles((prev) => {
      const next = [...prev];
      next[index] = p;
      return next;
    });
  }, []);

  const addPerson = useCallback(() => {
    setGroupProfiles((prev) =>
      prev.length < 5 ? [...prev, emptyProfile()] : prev
    );
  }, []);

  const removePerson = useCallback((index: number) => {
    setGroupProfiles((prev) =>
      prev.length > 2 ? prev.filter((_, i) => i !== index) : prev
    );
  }, []);

  const handleSoloSubmit = useCallback(async () => {
    setSoloLoading(true);
    setSoloError(null);
    setSoloSearched(true);
    try {
      const data = await fetchRecommendations(soloProfile, 10);
      setSoloResults(data);
    } catch (err: unknown) {
      setSoloError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setSoloLoading(false);
    }
  }, [soloProfile]);

  const handleGroupSubmit = useCallback(async () => {
    setGroupLoading(true);
    setGroupError(null);
    setGroupSearched(true);
    try {
      const data = await fetchGroupRecommendations(groupProfiles, 10);
      setGroupResults(data);
    } catch (err: unknown) {
      setGroupError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setGroupLoading(false);
    }
  }, [groupProfiles]);

  return (
    <div className={s.app}>
      <header className={s.header}>
        <h1 className={s.title}>DineOutBuddy</h1>
        <p className={s.subtitle}>
          Find the perfect restaurant for you — or your whole group
        </p>
      </header>

      <div className={s.modeTabs}>
        <button
          className={`${s.modeTab} ${mode === "solo" ? s.modeTabActive : ""}`}
          onClick={() => setMode("solo")}
        >
          Solo
        </button>
        <button
          className={`${s.modeTab} ${mode === "group" ? s.modeTabActive : ""}`}
          onClick={() => setMode("group")}
        >
          Group
        </button>
      </div>

      {mode === "solo" && (
        <>
          <PreferenceForm profile={soloProfile} onChange={setSoloProfile} />

          <div className={r.submitRow}>
            <button
              className={r.submitBtn}
              onClick={handleSoloSubmit}
              disabled={soloLoading}
            >
              {soloLoading ? "Searching..." : "Find Restaurants"}
            </button>
          </div>

          {soloError && <div className={r.error}>{soloError}</div>}

          {soloLoading && (
            <div className={r.loading}>
              <div className={r.spinner} />
              <div>Finding your perfect match...</div>
            </div>
          )}

          {!soloLoading && soloSearched && soloResults.length === 0 && !soloError && (
            <div className={r.empty}>
              <div className={r.emptyIcon}>:/</div>
              <div>No restaurants matched your criteria.</div>
              <div>Try adjusting your dietary requirements or cuisine preferences.</div>
            </div>
          )}

          {!soloLoading &&
            soloResults.map((rec) => (
              <RestaurantCard key={rec.restaurant_id} rec={rec} />
            ))}
        </>
      )}

      {mode === "group" && (
        <>
          {groupProfiles.map((profile, i) => (
            <PreferenceForm
              key={i}
              profile={profile}
              onChange={(p) => updateGroupProfile(i, p)}
              personIndex={i}
              onRemove={groupProfiles.length > 2 ? () => removePerson(i) : undefined}
            />
          ))}

          <div className={r.groupAddRow}>
            <button
              className={r.addPersonBtn}
              onClick={addPerson}
              disabled={groupProfiles.length >= 5}
            >
              + Add Person ({groupProfiles.length}/5)
            </button>
          </div>

          <div className={r.submitRow}>
            <button
              className={r.submitBtn}
              onClick={handleGroupSubmit}
              disabled={groupLoading}
            >
              {groupLoading ? "Searching..." : "Find Group Restaurants"}
            </button>
          </div>

          {groupError && <div className={r.error}>{groupError}</div>}

          {groupLoading && (
            <div className={r.loading}>
              <div className={r.spinner} />
              <div>Finding the best spot for everyone...</div>
            </div>
          )}

          {!groupLoading && groupSearched && groupResults.length === 0 && !groupError && (
            <div className={r.empty}>
              <div className={r.emptyIcon}>:/</div>
              <div>No restaurants work for everyone.</div>
              <div>Try relaxing some dietary requirements.</div>
            </div>
          )}

          {!groupLoading &&
            groupResults.map((rec) => (
              <GroupRestaurantCard
                key={rec.restaurant_id}
                rec={rec}
                numPeople={groupProfiles.length}
              />
            ))}
        </>
      )}
    </div>
  );
}
