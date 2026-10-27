import type { ExpressionSpecification } from "maplibre-gl";

// Harvest-date ramp: early in the zafra = light yellow, late = dark blue (viridis stops).
export const SEASON_START = "2025-10-27"; // first week of results/4e/weekly.csv
export const SEASON_END = "2026-05-31";
export const NO_DATE_COLOR = "#9a9a9a";

const RAMP = ["#fde725", "#5ec962", "#21918c", "#3b528b", "#440154"];

export function dayOfSeason(isoDate: string): number {
  return Math.round((Date.parse(isoDate) - Date.parse(SEASON_START)) / 86_400_000);
}

export const SEASON_DAYS = dayOfSeason(SEASON_END);

/** Fill colour from the numeric `harvest_day` property added on load (null = grey). */
export function harvestColorExpression(): ExpressionSpecification {
  const stops = RAMP.flatMap((c, i) => [(i * SEASON_DAYS) / (RAMP.length - 1), c]);
  return [
    "case",
    ["==", ["get", "harvest_day"], null],
    NO_DATE_COLOR,
    ["interpolate", ["linear"], ["get", "harvest_day"], ...stops],
  ] as ExpressionSpecification;
}

export const RAMP_CSS = `linear-gradient(to right, ${RAMP.join(", ")})`;
