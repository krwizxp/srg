const http = require("node:http");
const fs = require("node:fs");
const assert = require("node:assert/strict");
const { performance } = require("node:perf_hooks");
const { chromium, firefox, webkit } = require("playwright");
const { gzipSync } = require("node:zlib");

const source = {
  baseline: fs.readFileSync(".validation/revgeo-baseline.html"),
  candidate: fs.readFileSync(".validation/revgeo-candidate.html"),
};
const TARGET = 300;
const BROADCAST_ITERATIONS = 600;
const SAMPLE_ROUNDS = 6;
const INPUT = Array(TARGET).fill("대한민국 위경도: 37.5665,126.978").join("\n") + "\n";
const SOURCE = JSON.stringify({
  type: "FeatureCollection",
  features: [{properties: {geocoding: {
    label: "서울특별시 중구 세종대로 110",
    country_code: "kr",
    name: "서울시청",
    type: "public",
    admin: {level4: "서울특별시", level6: "중구"},
  }}}],
});
const app = http.createServer((req, res) => {
  const name = req.url.includes("/baseline/") ? "baseline" : "candidate";
  if (req.url.includes("revgeo.html")) {
    res.writeHead(200, {"content-type": "text/html; charset=utf-8", "cache-control": "no-store"});
    res.end(source[name]);
  } else if (req.url.includes("icon.svg")) {
    res.writeHead(200, {"content-type": "image/svg+xml"});
    res.end('<svg xmlns="http://www.w3.org/2000/svg"/>');
  } else {
    res.writeHead(404);
    res.end();
  }
});
const browserTypes = [["chromium", chromium], ["firefox", firefox], ["webkit", webkit]];
function stats(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return {
    median: +(sorted[Math.floor(sorted.length / 2) - 1] + sorted[Math.floor(sorted.length / 2)]) / 2,
    p95: sorted[Math.ceil(.95 * sorted.length) - 1],
    min: sorted[0],
    max: sorted[sorted.length - 1],
    samples: values.length,
  };
}
const measured = [];
async function fixture(page) {
  const time = await page.evaluate(() => performance.now());
  await page.locator("#file").setInputFiles({
    name: "coordinates.txt",
    mimeType: "text/plain",
    buffer: Buffer.from(INPUT),
  });
  await page.locator("#parse").click();
  await page.waitForFunction((total) => {
    const count = Number(document.getElementById("coords-count").textContent);
    const start = document.getElementById("start");
    return count === total && start.disabled === false;
  }, TARGET, {timeout: 15000});
  return await page.evaluate(() => performance.now()) - time;
}
async function finishLookup(page, n) {
  await page.waitForFunction((total) => {
    const start = document.getElementById("start");
    return Number(document.getElementById("cnt-success").textContent) === total
      && start.hidden === false
      && !start.disabled
      && document.getElementById("status").textContent.includes("조회 완료");
  }, n, {timeout: 18000});
}
async function lookup(page) {
  const time = await page.evaluate(() => performance.now());
  await page.locator("#start").click();
  await finishLookup(page, TARGET);
  return await page.evaluate(() => performance.now()) - time;
}
async function channel(page) {
  await page.evaluate((iterations) => {
    window.__broadcastMetric.count = 0;
    window.__broadcastMetric.totalMs = 0;
    const sender = new BroadcastChannel("srg-revgeo-cache:v4");
    for (let i = 0; i < iterations; i++) {
      const id = "bench-token-" + i;
      sender.postMessage([0, id]);
      sender.postMessage([1, id]);
    }
    sender.close();
  }, BROADCAST_ITERATIONS);
  await page.waitForFunction((expected) =>
    window.__broadcastMetric.count >= expected, BROADCAST_ITERATIONS * 2, {timeout:15000});
  return await page.evaluate(() => ({
    count: window.__broadcastMetric.count,
    totalMs: window.__broadcastMetric.totalMs,
  }));
}
async function trial(browser, type, version, port, round) {
  const context = await browser.newContext({
    viewport: {width: 1280, height: 850},
    serviceWorkers: "block",
  });
  let requests = 0;
  const errors = [];
  try {
    await context.route("https://nominatim.openstreetmap.org/**", async route => {
      requests++;
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        headers: {"access-control-allow-origin": "*"},
        body: SOURCE,
      });
    });
    const page = await context.newPage();
    page.on("pageerror", err => errors.push(err.message));
    await page.addInitScript(() => {
      window.__broadcastMetric = {count: 0, totalMs: 0};
      const Original = window.BroadcastChannel;
      window.BroadcastChannel = class extends Original {
        set onmessage(callback) {
          this._ownOnMessage = callback;
          super.onmessage = event => {
            const now = performance.now();
            try {
              return callback(event);
            } finally {
              window.__broadcastMetric.count++;
              window.__broadcastMetric.totalMs += performance.now() - now;
            }
          };
        }
        get onmessage() { return this._ownOnMessage ?? null; }
      };
    });
    const url = "http://127.0.0.1:" + port + "/" + version + "/revgeo.html";
    await page.goto(url, {waitUntil: "domcontentloaded"});
    const navigation = await page.evaluate(() => {
      const timing = performance.getEntriesByType("navigation")[0];
      return timing ? timing.domContentLoadedEventEnd : null;
    });
    const parse = await fixture(page);
    const cold = await lookup(page);
    assert.equal(requests, 1, "First query should contact mocked Nominatim exactly once");
    const warm = await lookup(page);
    assert.equal(requests, 1, "Memory cache must eliminate all subsequent requests");
    const secondPage = await context.newPage();
    secondPage.on("pageerror", err => errors.push(err.message));
    await secondPage.goto(url, {waitUntil: "domcontentloaded"});
    await fixture(secondPage);
    const disk = await lookup(secondPage);
    assert.equal(requests, 1, "Second tab must reuse real persisted IndexedDB cache");
    const events = await channel(page);
    assert.equal(events.count, BROADCAST_ITERATIONS * 2, "All cross-tab cache messages must be delivered");
    const heap = await page.evaluate(() => performance.memory?.usedJSHeapSize ?? null);
    assert.deepEqual(errors, [], "No real browser runtime errors");
    const outcome = {
      browser: type, version, round, navigation, parse, cold, warm,
      disk, broadcastMs: events.totalMs, heapBytes: heap,
      lookups: TARGET, requests, cacheMessages: events.count,
    };
    console.log(JSON.stringify(outcome));
    return outcome;
  } finally {
    await context.close();
  }
}
function verify(rows) {
  const summary = {};
  for (const [browser] of browserTypes) {
    summary[browser] = {};
    for (const mode of ["baseline", "candidate"]) {
      const slice = rows.filter(r => r.browser === browser && r.version === mode);
      assert.equal(slice.length, SAMPLE_ROUNDS);
      summary[browser][mode] = Object.fromEntries(
        ["navigation", "parse", "cold", "warm", "disk", "broadcastMs", "heapBytes"]
          .map(key => [key, slice.every(row => row[key] != null)
            ? stats(slice.map(row => row[key])) : null]));
    }
    for (const key of ["navigation", "parse", "cold", "warm", "disk"]) {
      const base = summary[browser].baseline[key];
      const candidate = summary[browser].candidate[key];
      const permissibleMedian = Math.max(base.median * 1.20, base.median + 35);
      const permissibleTail = Math.max(base.p95 * 1.30, base.p95 + 80);
      assert(candidate.median <= permissibleMedian,
        browser + " " + key + " median performance regression: " + JSON.stringify({base,candidate,permissibleMedian}));
      assert(candidate.p95 <= permissibleTail,
        browser + " " + key + " tail performance regression: " + JSON.stringify({base,candidate,permissibleTail}));
    }
    const before = summary[browser].baseline.broadcastMs;
    const after = summary[browser].candidate.broadcastMs;
    assert(after.median <= Math.max(before.median * 1.4, before.median + 2),
      "BroadcastChannel callback performance degraded: " + JSON.stringify({browser,before,after}));
    if (summary[browser].baseline.heapBytes && summary[browser].candidate.heapBytes) {
      const beforeHeap = summary[browser].baseline.heapBytes;
      const afterHeap = summary[browser].candidate.heapBytes;
      assert(afterHeap.median <= beforeHeap.median * 1.10 + 3_000_000,
        "Retained browser JS heap exceeds controlled nonregression margin: " + JSON.stringify({browser,beforeHeap,afterHeap}));
    }
  }
  return summary;
}
async function run() {
  await new Promise(resolve => app.listen(0, "127.0.0.1", resolve));
  const port = app.address().port;
  try {
    const b = source.baseline, c = source.candidate;
    console.log("SOURCE_BENCH_INPUT", JSON.stringify({
      sourceBaseline: process.env.SRG_BASELINE_SHA,
      sourceCandidate: process.env.SRG_CANDIDATE_SHA,
      baselineBytes: b.length, candidateBytes: c.length,
      rawDiffBytes: c.length - b.length,
      baselineGzipBytes: gzipSync(b).length,
      candidateGzipBytes: gzipSync(c).length,
      sampleRounds: SAMPLE_ROUNDS,
      lookupCount: TARGET,
      validCrossTabMessages: BROADCAST_ITERATIONS * 2,
      browserVersions: {},
    }));
    for (const [name, engine] of browserTypes) {
      const browser = await engine.launch({headless: true});
      try {
        console.log("BROWSER_VERSION",name,browser.version());
        for (let n = 0; n < SAMPLE_ROUNDS; n++) {
          const order = n % 2 === 0 ? ["baseline", "candidate"] : ["candidate", "baseline"];
          for (const mode of order) {
            measured.push(await trial(browser,name,mode,port,n));
          }
        }
      } finally {
        await browser.close();
      }
    }
    const summary = verify(measured);
    const report = {
      sourceBaseline: process.env.SRG_BASELINE_SHA,
      sourceCandidate: process.env.SRG_CANDIDATE_SHA,
      sourceBytes: {baseline: b.length,candidate:c.length},
      sampleRounds:SAMPLE_ROUNDS,
      cases:measured,
      summary,
    };
    fs.writeFileSync(".validation/perf-results.json",JSON.stringify(report,null,2));
    for (const [name, record] of Object.entries(summary)) {
      for (const key of ["navigation","parse","cold","warm","disk","broadcastMs","heapBytes"]) {
        const before=record.baseline[key],after=record.candidate[key];
        if(before && after) console.log("COMPARE",name,key,
          "median",before.median,"->",after.median,
          "p95",before.p95,"->",after.p95);
      }
    }
    console.log("BASELINE_CANDIDATE_BROWSER_PERF: PASS");
  } finally {
    await new Promise(resolve => app.close(resolve));
  }
}
run().catch(err => {
  console.error(err);
  fs.writeFileSync(".validation/perf-results.json",JSON.stringify({cases:measured,error:String(err)},null,2));
  process.exitCode = 1;
});
