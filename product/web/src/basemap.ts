// Basemap with an offline fallback. The style always has a plain background colour; the
// Esri World Imagery raster is shown only after a probe tile loads, and hidden again if
// tiles start failing (wifi drops). The sample polygons never depend on the network.

export const BASEMAP_SOURCE = "esri-imagery";
export const BASEMAP_LAYER = "basemap";
export const ESRI_TILES =
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
export const ESRI_ATTRIBUTION =
  "Imagery © Esri, Maxar, Earthstar Geographics, and the GIS User Community";

export type BasemapState = "checking" | "online" | "offline";

/** True if one Esri tile answers within `timeoutMs`. `no-cors` gives an opaque response,
 * which is enough: a network failure rejects, a reachable server resolves. */
export async function probeBasemap(timeoutMs = 3000): Promise<boolean> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const url = ESRI_TILES.replace("{z}", "1").replace("{y}", "0").replace("{x}", "0");
    await fetch(url, { mode: "no-cors", cache: "no-store", signal: ctrl.signal });
    return true;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}
