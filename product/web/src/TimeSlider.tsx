import { useState } from "react";
import { SEASON_DAYS, SEASON_START } from "./colors";

const WEEKS = Math.ceil(SEASON_DAYS / 7);

function weekStart(week: number): string {
  const d = new Date(Date.parse(SEASON_START) + week * 7 * 86_400_000);
  return d.toISOString().slice(0, 10);
}

// Placeholder: the value is only logged. Filtering the map by harvest_date is a hackathon item.
export default function TimeSlider() {
  const [week, setWeek] = useState(WEEKS);
  return (
    <div className="slider" data-testid="slider">
      <label htmlFor="week">Week of {weekStart(week)}</label>
      <input
        id="week"
        type="range"
        min={0}
        max={WEEKS}
        value={week}
        onChange={(e) => {
          const w = Number(e.target.value);
          setWeek(w);
          console.log("[slider]", { week: w, weekStart: weekStart(w) });
        }}
      />
    </div>
  );
}
