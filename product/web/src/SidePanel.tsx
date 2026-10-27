import type { FieldProps } from "./App";
import { NO_DATE_COLOR, RAMP_CSS, SEASON_END, SEASON_START } from "./colors";

// Placeholder panel: shows the clicked polygon's properties. The detail view (facts,
// summary, HV/coherence chart, alerts) is a hackathon item.
export default function SidePanel({ selected }: { selected: FieldProps | null }) {
  return (
    <aside className="panel" data-testid="panel">
      <section>
        <h2>Harvest date</h2>
        <div className="ramp" style={{ background: RAMP_CSS }} />
        <div className="ramp-labels">
          <span>{SEASON_START}</span>
          <span>{SEASON_END}</span>
        </div>
        <div className="nodate">
          <span className="swatch" style={{ background: NO_DATE_COLOR }} /> no date
        </div>
      </section>
      <section>
        <h2>Field</h2>
        {selected ? (
          <table className="props">
            <tbody>
              {Object.entries(selected).map(([k, v]) => (
                <tr key={k}>
                  <th>{k}</th>
                  <td>{String(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="muted">Click a polygon on the map.</p>
        )}
      </section>
      <section>
        <p className="muted small">Sample: 50 Morelos segments, 2025-26 zafra. Placeholder panel.</p>
      </section>
    </aside>
  );
}
