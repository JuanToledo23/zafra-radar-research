import type { Feature, FeatureCollection, MultiPolygon, Polygon } from "geojson";
import maplibregl, { type StyleSpecification } from "maplibre-gl";
import { useEffect, useRef } from "react";
import type { FieldProps } from "./App";
import {
  BASEMAP_LAYER,
  BASEMAP_SOURCE,
  ESRI_ATTRIBUTION,
  ESRI_TILES,
  probeBasemap,
  type BasemapState,
} from "./basemap";
import { dayOfSeason, harvestColorExpression } from "./colors";

// No glyphs or sprites: the style needs nothing from the network to render.
const STYLE: StyleSpecification = {
  version: 8,
  sources: {
    [BASEMAP_SOURCE]: {
      type: "raster",
      tiles: [ESRI_TILES],
      tileSize: 256,
      maxzoom: 19,
      attribution: ESRI_ATTRIBUTION,
    },
  },
  layers: [
    { id: "background", type: "background", paint: { "background-color": "#e9e5da" } },
    { id: BASEMAP_LAYER, type: "raster", source: BASEMAP_SOURCE, layout: { visibility: "none" } },
  ],
};

type Props = {
  onSelect: (p: FieldProps | null) => void;
  onBasemap: (s: BasemapState) => void;
};

export default function MapView({ onSelect, onBasemap }: Props) {
  const container = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const map = new maplibregl.Map({
      container: container.current!,
      style: STYLE,
      center: [-99.176, 18.638],
      zoom: 14,
      attributionControl: { compact: true },
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-left");
    map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");

    const setBasemap = (s: BasemapState) => {
      if (!map.getLayer(BASEMAP_LAYER)) return;
      map.setLayoutProperty(BASEMAP_LAYER, "visibility", s === "online" ? "visible" : "none");
      onBasemap(s);
    };

    // Tile failures after going online (e.g. wifi off mid-session) fall back to plain.
    map.on("error", (e) => {
      if ((e as { sourceId?: string }).sourceId === BASEMAP_SOURCE) setBasemap("offline");
    });

    map.on("load", async () => {
      const fc: FeatureCollection = await fetch("/sample.geojson").then((r) => r.json());
      const bounds = new maplibregl.LngLatBounds();
      fc.features.forEach((f: Feature, i: number) => {
        const p = f.properties ?? {};
        p.id = i;
        p.harvest_day = p.harvest_date ? dayOfSeason(p.harvest_date) : null;
        f.properties = p;
        const geom = f.geometry as MultiPolygon | Polygon;
        const rings = geom.type === "Polygon" ? [geom.coordinates] : geom.coordinates;
        rings.forEach((poly) => poly[0].forEach(([x, y]) => bounds.extend([x, y])));
      });

      map.addSource("sample", { type: "geojson", data: fc, promoteId: "id" });
      map.addLayer({
        id: "fields-fill",
        type: "fill",
        source: "sample",
        paint: { "fill-color": harvestColorExpression(), "fill-opacity": 0.75 },
      });
      map.addLayer({
        id: "fields-line",
        type: "line",
        source: "sample",
        paint: {
          "line-color": ["case", ["boolean", ["feature-state", "selected"], false], "#d7263d", "#1d1d1d"],
          "line-width": ["case", ["boolean", ["feature-state", "selected"], false], 3, 0.6],
        },
      });
      map.fitBounds(bounds, { padding: 60, duration: 0 });

      let selectedId: number | null = null;
      map.on("click", (e) => {
        const hit = map.queryRenderedFeatures(e.point, { layers: ["fields-fill"] })[0];
        if (selectedId !== null) map.setFeatureState({ source: "sample", id: selectedId }, { selected: false });
        selectedId = hit ? (hit.id as number) : null;
        if (selectedId !== null) map.setFeatureState({ source: "sample", id: selectedId }, { selected: true });
        const props = hit ? { ...(hit.properties as FieldProps) } : null;
        if (props) {
          delete props.harvest_day;
          delete props.id;
        }
        onSelect(props);
      });
      map.on("mouseenter", "fields-fill", () => (map.getCanvas().style.cursor = "pointer"));
      map.on("mouseleave", "fields-fill", () => (map.getCanvas().style.cursor = ""));

      setBasemap((await probeBasemap()) ? "online" : "offline");
      // Read by the headless screenshot script (product/scripts/screenshot.mjs).
      map.once("idle", () => Object.assign(window, { __mapReady: true, __zafraMap: map }));
    });

    return () => map.remove();
  }, [onSelect, onBasemap]);

  return <div ref={container} className="map" data-testid="map" />;
}
