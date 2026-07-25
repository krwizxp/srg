import { chromium, firefox, webkit } from "playwright";
import { createServer } from "node:http";
import { readFile, writeFile } from "node:fs/promises";
import { extname, resolve } from "node:path";
const root = process.cwd();
const contentTypes = new Map([[".html", "text/html; charset=utf-8"], [".svg", "image/svg+xml"]]);
const server = createServer(async (req, res) => {
  try {
    const path = resolve(root, `.${new URL(req.url, "http://localhost").pathname}`);
    if (!path.startsWith(root)) throw new Error("outside root");
    const body = await readFile(path);
    res.writeHead(200, { "content-type": contentTypes.get(extname(path)) ?? "application/octet-stream" });
    res.end(body);
  } catch {
    res.writeHead(404);
    res.end();
  }
});
await new Promise((resolveListen) => server.listen(0, "127.0.0.1", resolveListen));
const { port } = server.address();
const url = `http://127.0.0.1:${port}/revgeo.html`;
const coordinateText = "대한민국 위경도: 36.123456, 127.654321\n세계 위경도: -12.5, 140.25\n";
const expectedManual = `64비트 난수: 0 (유부호 정수: 0)
2진수: 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000
8진수: 0
16진수: 00 00 00 00 00 00 00 00
Hex 코드: #000000 #000000
바이트 배열: 0 0 0 0 0 0 0 0
6자리 숫자 비밀번호: 000000
8자리 비밀번호: !!!!!!!!
로또 번호: 1 2 3 4 5 6
일본 로또 7 번호: 1 2 3 4 5 6 7
유로밀리언 번호: 1 2 3 4 5 + 1 9
한글 음절 4글자: 가가가가
대한민국 위경도: 33.1125, 124.609722
세계 위경도: -90, -180
NMS 은하 번호: 1
NMS 포탈 주소: 1 001 00 000 000 (🐦🌅🌅🐦🌅🌅🌅🌅🌅🌅🌅🌅)
NMS 은하 좌표: 07FF:007F:07FF:0001`;
const engines = { chromium, firefox, webkit };
const results = [];
for (const [engineName, engine] of Object.entries(engines)) {
  const browser = await engine.launch({ headless: true });
  const page = await browser.newPage({ acceptDownloads: true });
  const errors = [];
  page.on("pageerror", (error) => errors.push(String(error)));
  await page.route("https://nominatim.openstreetmap.org/**", async (route) => {
    const requestUrl = new URL(route.request().url());
    const lat = requestUrl.searchParams.get("lat");
    const lon = requestUrl.searchParams.get("lon");
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        type: "city",
        name: "테스트 지점",
        display_name: `테스트 주소 ${lat},${lon}`,
        address: { country: "대한민국", country_code: "kr", state: "충청남도", city: "천안시", suburb: "불당동" },
      }),
    });
  });
  await page.goto(url, { waitUntil: "domcontentloaded" });
  const capabilities = await page.evaluate(() => ({
    commandEvent: typeof CommandEvent,
    commandFor: "commandForElement" in HTMLButtonElement.prototype,
    promiseTry: typeof Promise.try,
    promiseWithResolvers: typeof Promise.withResolvers,
    startViewTransition: typeof document.startViewTransition,
  }));
  await page.locator(".manual-rng").evaluate((element) => { element.open = true; });
  await page.locator("#num64").fill("0");
  await page.locator("#suppValues").fill("1 2 3 4 5 6 7");
  await page.locator("#runBtn").click();
  await page.waitForFunction(() => document.querySelector("#meta")?.textContent.includes("완료"));
  const manualOutput = await page.locator("#output .coords-content").innerText();
  await page.locator("#file").setInputFiles({ name: "coords.txt", mimeType: "text/plain", buffer: Buffer.from(coordinateText) });
  await page.locator("#parse").click();
  await page.waitForFunction(() => document.querySelector("#status")?.textContent.includes("분석 완료"));
  const coords = await page.locator("#coords .coords-content").innerText();
  await page.locator("#start").click();
  await page.waitForFunction(() => document.querySelector("#status")?.textContent.includes("모든 처리가 완료"), null, { timeout: 20000 });
  const processed = await page.locator("#cnt-success").innerText();
  const rowCount = await page.locator("#results tr").count();
  await page.locator("#filter-q").fill("테스트 주소");
  await page.waitForTimeout(700);
  const visibleRows = await page.locator("#results tr:not([hidden])").count();
  const oldTheme = await page.locator("html").getAttribute("data-theme");
  await page.locator("#theme-toggle").click();
  const newTheme = await page.locator("html").getAttribute("data-theme");
  const ids = await page.locator("[id]").evaluateAll((elements) => elements.map((element) => element.id));
  const duplicateIds = [...new Set(ids.filter((id, index) => ids.indexOf(id) !== index))];
  const missingAriaRefs = await page.evaluate(() => {
    const missing = [];
    for (const element of document.querySelectorAll("*")) {
      for (const attr of ["aria-controls", "aria-describedby", "aria-labelledby"]) {
        const value = element.getAttribute(attr);
        if (!value) continue;
        for (const id of value.split(/\s+/)) {
          if (!document.getElementById(id)) missing.push([element.id || element.tagName, attr, id]);
        }
      }
    }
    return missing;
  });
  results.push({
    engine: engineName,
    capabilities,
    errors,
    manualMatchesRust: manualOutput === expectedManual,
    coords,
    processed,
    rowCount,
    visibleRows,
    themeChanged: oldTheme !== newTheme,
    duplicateIds,
    missingAriaRefs,
  });
  await browser.close();
}
server.close();
await writeFile("browser-results.json", JSON.stringify({ playwrightVersion: process.env.npm_package_dependencies_playwright ?? "latest", results }, null, 2));
for (const result of results) {
  if (result.errors.length || !result.manualMatchesRust || result.coords !== "36.123456, 127.654321\n-12.5, 140.25" || result.processed !== "2" || result.rowCount !== 2 || result.visibleRows !== 2 || !result.themeChanged || result.duplicateIds.length || result.missingAriaRefs.length) {
    throw new Error(`${result.engine} verification failed: ${JSON.stringify(result)}`);
  }
}
