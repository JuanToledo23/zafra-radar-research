import { useEffect, useState } from "react";
import MapView from "./MapView";
import SidePanel from "./SidePanel";
import TimeSlider from "./TimeSlider";
import type { BasemapState } from "./basemap";

export type FieldProps = Record<string, string | number | boolean | null>;

export default function App() {
  const [selected, setSelected] = useState<FieldProps | null>(null);
  const [basemap, setBasemap] = useState<BasemapState>("checking");
  const [api, setApi] = useState<string>("checking");

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((h) => setApi(h.status === "ok" ? "ok" : `degraded (db ${h.db})`))
      .catch(() => setApi("unreachable"));
  }, []);

  return (
    <div className="app">
      <header className="header">
        <h1>Zafra Radar</h1>
        <div className="badges">
          <span className={`badge badge-${basemap}`} data-testid="basemap-state">
            basemap: {basemap === "online" ? "satellite" : basemap === "offline" ? "offline (plain)" : "checking"}
          </span>
          <span className={`badge badge-${api === "ok" ? "online" : "offline"}`} data-testid="api-state">
            api: {api}
          </span>
        </div>
      </header>
      <main className="main">
        <MapView onSelect={setSelected} onBasemap={setBasemap} />
        <SidePanel selected={selected} />
        <TimeSlider />
      </main>
    </div>
  );
}
