const assert = require("node:assert/strict");
const { createHash } = require("node:crypto");
const { chromium, firefox, webkit } = require("playwright");

// Fixed before execution: 31 alternating pairs, independent cold pages and
// warmed pages, 16/1000 records, 5% one-sided median-ratio bootstrap bound.
const PAIRS = 31;
const LIMIT = 1.05;
let state = 0x9e3779b97f4a7c15n;
function next() {
  state = (state * 6364136223846793005n + 1442695040888963407n) & 0xffffffffffffffffn;
  return state.toString();
}
const inputs = ["0", "1", "18446744073709551615", ...Array.from({length: 997}, next)];
const supplements = Array.from({length: 12000}, next).join("\n");
const median = values => [...values].sort((a,b) => a-b)[values.length >> 1];
function upperBound(before, after) {
  const ratios = before.map((value, index) => after[index] / value);
  let seed = 20260919;
  const samples = [];
  for (let iteration = 0; iteration < 10000; iteration++) {
    const picked = [];
    for (let index = 0; index < ratios.length; index++) {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      picked.push(ratios[seed % ratios.length]);
    }
    samples.push(median(picked));
  }
  samples.sort((a,b) => a-b);
  return samples[9500];
}
async function openPage(context, port, count) {
  const page = await context.newPage();
  page.setDefaultTimeout(30000);
  await page.goto(`http://127.0.0.1:${port}/revgeo.html`, {waitUntil:"domcontentloaded"});
  // Fixture preparation is excluded from timing. Assign the large buffers
  // directly instead of exercising the automation paste/clipboard path.
  const raw = inputs.slice(0,count).join("\n");
  await page.evaluate(([raw, supplements]) => {
    document.querySelector("#num64").value = raw;
    document.querySelector("#suppValues").value = supplements;
  }, [raw, supplements]);
  assert.equal(await page.locator("#num64").inputValue(), raw);
  assert.equal(await page.locator("#suppValues").inputValue(), supplements);
  await page.locator("#runBtn").scrollIntoViewIfNeeded();
  return page;
}
async function generate(page, count) {
  const elapsed = await page.evaluate(count => new Promise((resolve, reject) => {
    const meta = document.querySelector("#meta");
    const error = document.querySelector("#error");
    const started = performance.now();
    const timer = setTimeout(() => {
      observer.disconnect();
      reject(new Error("generation timed out: " + meta.textContent + " / " + error.textContent));
    }, 30000);
    const observer = new MutationObserver(() => {
      if (error.textContent) {
        clearTimeout(timer);
        observer.disconnect();
        reject(new Error(error.textContent));
      } else if (meta.textContent.startsWith(`기본 난수 변환 완료: ${count.toLocaleString("ko-KR")}개`)) {
        const elapsed = performance.now() - started;
        clearTimeout(timer);
        observer.disconnect();
        resolve(elapsed);
      }
    });
    observer.observe(meta, {childList:true,subtree:true,characterData:true});
    observer.observe(error, {childList:true,subtree:true,characterData:true});
    document.querySelector("#runBtn").click();
  }), count);
  const output = await page.locator("#output .coords-content").textContent();
  assert.equal(await page.locator("#saveOutput").isEnabled(), true);
  return {elapsed, hash:createHash("sha256").update(output).digest("hex")};
}
async function errorsAndCancellation(context, port) {
  const page = await openPage(context, port, 16);
  try {
    const observations = [];
    for (const invalid of ["-1", "18446744073709551616", "not-a-number"]) {
      await page.locator("#num64").fill(invalid);
      await page.locator("#runBtn").click();
      observations.push(await page.locator("#error").textContent());
      assert.ok(observations.at(-1));
      assert.equal(await page.locator("#saveOutput").isEnabled(), false);
    }
    await page.locator("#num64").fill("0");
    await page.locator("#suppValues").fill("");
    await page.locator("#runBtn").click();
    await page.locator("#supp-dialog").waitFor({state:"visible"});
    await page.locator("#supp-dialog-values").press("Escape");
    await page.waitForFunction(() => document.querySelector("#meta").textContent.includes("중단"));
    observations.push(await page.locator("#meta").textContent());
    assert.equal(await page.locator("#saveOutput").isEnabled(), false);
    assert.equal(await page.locator("#clearBtn").evaluate(node => node === document.activeElement), true);
    await page.locator("#clearBtn").click();
    assert.equal(await page.locator("#meta").textContent(), "");
    return observations;
  } finally { await page.close(); }
}

(async () => {
  const results = [];
  for (const [engine, browserType] of Object.entries({chromium,firefox,webkit})) {
    const browser = await browserType.launch({headless:true});
    try {
      const context = await browser.newContext({locale:"ko-KR"});
      // External lookups/fonts are excluded from local UI timing on both sides.
      await context.route("https://**", route => route.abort());
      assert.deepEqual(await errorsAndCancellation(context,4173), await errorsAndCancellation(context,4174));
      for (const count of [16,1000]) {
        for (const mode of ["cold","warm"]) {
          let pages = null;
          if (mode === "warm") {
            pages = [await openPage(context,4173,count),await openPage(context,4174,count)];
            for (let i=0;i<8;i++) for (const page of pages) await generate(page,count);
          }
          const before = [], after = [];
          for (let pair=0;pair<PAIRS;pair++) {
            const measured = [];
            for (const side of pair%2 ? [1,0] : [0,1]) {
              const page = pages ? pages[side] : await openPage(context,4173+side,count);
              try { measured[side] = await generate(page,count); }
              finally { if (!pages) await page.close(); }
            }
            assert.equal(measured[0].hash,measured[1].hash);
            before.push(measured[0].elapsed);
            after.push(measured[1].elapsed);
          }
          if (pages) for (const page of pages) await page.close();
          const upper = upperBound(before,after);
          const result = {engine,version:browser.version(),count,mode,pairs:PAIRS,
            before_ms:before,after_ms:after,median_before:median(before),median_after:median(after),
            upper95_ratio:upper,limit:LIMIT,pass:upper<=LIMIT};
          results.push(result);
          console.log("RESULT " + JSON.stringify(result));
        }
      }
    } finally { await browser.close(); }
  }
  console.log("SUMMARY " + JSON.stringify({comparisons:results.length,passed:results.filter(x=>x.pass).length}));
  assert.ok(results.every(x=>x.pass),"browser noninferiority bound exceeded; reject or investigate candidate");
})().catch(error => { console.error(error); process.exitCode=1; });
