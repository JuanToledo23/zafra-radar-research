// Headless check of the map shell: load the page, wait for the map to go idle, click the
// polygon nearest the map centre, move the slider, save a screenshot and print a
// JSON report. --offline aborts every request that is not to localhost, as with wifi off.
//   node screenshot.mjs [--url http://localhost:5174] [--out ../docs/skeleton.png] [--offline]
import { chromium } from "playwright";

const arg = (name, dflt) => {
  const i = process.argv.indexOf(name);
  return i > -1 ? process.argv[i + 1] : dflt;
};
const url = arg("--url", `http://localhost:${process.env.WEB_PORT ?? "5174"}`);
const out = arg("--out", new URL("../docs/skeleton.png", import.meta.url).pathname);
const offline = process.argv.includes("--offline");

const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
const blocked = [];
const errors = [];
const logs = [];
if (offline) {
  await page.route("**/*", (route) => {
    const host = new URL(route.request().url()).hostname;
    if (host === "localhost" || host === "127.0.0.1") return route.continue();
    blocked.push(host);
    return route.abort("internetdisconnected");
  });
}
page.on("pageerror", (e) => errors.push(String(e)));
page.on("console", (m) => logs.push(`${m.type()}: ${m.text()}`));

const t0 = Date.now();
await page.goto(url);
await page.waitForFunction(() => window.__mapReady === true, null, { timeout: 30_000 });
const readyMs = Date.now() - t0;

const box = await page.getByTestId("map").boundingBox();
const hit = await page.evaluate(() => {
  const map = window.__zafraMap;
  const { width, height } = map.getCanvas().getBoundingClientRect();
  let best = null;
  for (let x = 10; x < width; x += 20) {
    for (let y = 10; y < height; y += 20) {
      if (!map.queryRenderedFeatures([x, y], { layers: ["fields-fill"] }).length) continue;
      const d = (x - width / 2) ** 2 + (y - height / 2) ** 2;
      if (!best || d < best.d) best = { x, y, d };
    }
  }
  return best;
});
if (hit) await page.mouse.click(box.x + hit.x, box.y + hit.y);
await page.getByTestId("panel").locator("table").waitFor({ timeout: 5_000 }).catch(() => {});
await page.locator("#week").fill("10");
await page.waitForTimeout(300);
await page.screenshot({ path: out });

const report = {
  url,
  offline,
  map_ready_ms: readyMs,
  basemap: await page.getByTestId("basemap-state").innerText(),
  api: await page.getByTestId("api-state").innerText(),
  panel: (await page.getByTestId("panel").locator("table").innerText().catch(() => "")).replace(/\s+/g, " "),
  slider_logs: logs.filter((l) => l.includes("[slider]")).length,
  blocked_hosts: [...new Set(blocked)],
  page_errors: errors,
  console_errors: logs.filter((l) => l.startsWith("error")),
  screenshot: out,
};
console.log(JSON.stringify(report, null, 2));
await browser.close();
