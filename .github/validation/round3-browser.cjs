const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium, firefox, webkit } = require("playwright");
const rustReference = fs.readFileSync(path.join(process.env.RUNNER_TEMP, "goal-round3-srg", "expected-browser.txt"), "utf8");

const engines = { chromium, firefox, webkit };
const coordinateFile = [
  "대한민국 위경도: 37.123456, 127.654321",
  "세계 위경도: -33.865143, 151.209900",
].join("\n");
let supplementState = 0x9e3779b97f4a7c15n;
const supplements = Array.from({ length: 4096 }, () => {
  supplementState = (
    supplementState * 6364136223846793005n + 1442695040888963407n
  ) & 0xffffffffffffffffn;
  return supplementState.toString();
}).join("\n");

async function waitForText(page, selector, pattern) {
  await page.waitForFunction(
    ([target, source, flags]) => new RegExp(source, flags)
      .test(document.querySelector(target)?.textContent || ""),
    [selector, pattern.source, pattern.flags],
  );
}

async function exercise(browserType, baseUrl) {
  const browser = await browserType.launch({ headless: true });
  try {
  const context = await browser.newContext({ locale: "ko-KR" });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", error => errors.push(`page: ${error.message}`));
  page.on("console", message => {
    if (message.type() === "error") errors.push(`console: ${message.text()} ${message.location().url}`);
  });
  await page.goto(`${baseUrl}/revgeo.html`, { waitUntil: "load" });
  assert.equal(await page.title(), "SRG 데이터 역지오코딩 도구");

  const themeBefore = await page.locator("html").getAttribute("data-theme");
  await page.locator("#theme-toggle").click();
  const themeAfter = await page.locator("html").getAttribute("data-theme");
  assert.notEqual(themeAfter, themeBefore);

  await page.locator("#file").setInputFiles({
    name: "random_data.txt",
    mimeType: "text/plain",
    buffer: Buffer.from(coordinateFile),
  });
  await page.locator("#parse").click();
  await waitForText(page, "#coords-count", /^2$/u);
  assert.equal(
    await page.locator("#coords .coords-content").textContent(),
    "37.123456, 127.654321\n-33.865143, 151.209900",
  );
  assert.equal(await page.locator("#coord-line-numbers").textContent(), "1\n2");

  await page.locator("#num64").fill("0\n1\n18446744073709551615");
  await page.evaluate(value => { document.querySelector("#suppValues").value = value; }, supplements);
  assert.equal(await page.locator("#suppValues").inputValue(), supplements);
  await page.locator("#runBtn").click();
  await waitForText(page, "#meta", /기본 난수 변환 완료: 3개/u);
  const manualOutput = await page.locator("#output .coords-content").textContent();
  assert.equal(manualOutput, rustReference, "Rust/browser shared conversion output differs");
  assert.match(manualOutput, /64비트 난수: 0 /u);
  assert.match(manualOutput, /64비트 난수: 18446744073709551615 /u);
  assert.ok(manualOutput.split("\n").length > 20);
  assert.equal(await page.locator("#saveOutput").isEnabled(), true);
  assert.equal(await page.locator("#copyOutput").isEnabled(), true);

  for (const invalid of ["-1", "18446744073709551616", "not-a-number"]) {
    await page.locator("#num64").fill(invalid);
    await page.locator("#runBtn").click();
    assert.ok(await page.locator("#error").textContent());
    assert.equal(await page.locator("#saveOutput").isEnabled(), false);
  }
  await page.locator("#num64").fill("0");
  await page.locator("#suppValues").fill("");
  await page.locator("#runBtn").click();
  await page.locator("#supp-dialog").waitFor({state: "visible"});
  await page.locator("#supp-dialog-values").press("Escape");
  await waitForText(page, "#meta", /중단/u);
  assert.equal(await page.locator("#clearBtn").evaluate(node => node === document.activeElement), true);
  await page.locator("#clearBtn").click();
  assert.equal(await page.locator("#meta").textContent(), "");
  process.stdout.write("BROWSER " + JSON.stringify({version: browser.version(), baseUrl, errors: errors.length}) + "\n");
  await page.setViewportSize({ width: 390, height: 844 });
  const horizontalOverflow = await page.evaluate(() =>
    document.documentElement.scrollWidth - window.innerWidth,
  );
  assert.ok(horizontalOverflow <= 1, `horizontal overflow: ${horizontalOverflow}`);
  assert.deepEqual(errors, []);
  return manualOutput;
  } finally {
    await browser.close();
  }
}

(async () => {
  for (const [name, browserType] of Object.entries(engines)) {
    const baseline = await exercise(browserType, "http://127.0.0.1:4173");
    const candidate = await exercise(browserType, "http://127.0.0.1:4174");
    assert.equal(candidate, baseline, `${name} manual output mismatch`);
    process.stdout.write(`${name}: behavior equivalent\n`);
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
