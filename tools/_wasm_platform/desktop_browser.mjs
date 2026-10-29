import fs from "node:fs/promises";
import path from "node:path";
import {pathToFileURL} from "node:url";
import {createRequire} from "node:module";

const [url, fixturesPath, outputDirectory, wasmSha256, boundaryPath = "./korea_boundary.js?v=6",
    engine = "chromium", channel] = process.argv.slice(2);
if (!url || !fixturesPath || !outputDirectory) throw new Error("Usage: desktop_browser.mjs URL FIXTURES OUTPUT WASM_SHA256 BOUNDARY_PATH ENGINE [CHANNEL]");
await fs.mkdir(outputDirectory, {recursive: true});
const require = createRequire(import.meta.url);
const playwrightPath = process.env.SRG_PLAYWRIGHT_MODULE || require.resolve("playwright");
const playwright = await import(pathToFileURL(playwrightPath).href);
const source = await fs.readFile(new URL("./browser_contract.js", import.meta.url), "utf8");
const fixtures = JSON.parse(await fs.readFile(fixturesPath, "utf8"));
const config = {fixtures, wasm: Boolean(wasmSha256), wasmSha256,
    wasmPath: "./coordinates.wasm", boundaryPath};
let browser;
let context;
let report;
const errors = [];
const consoleErrors = [];
const metadata = {environment: "Actual desktop browser; no mobile device emulation", engine, channel: channel ?? null};
try {
    const launch = {headless: true};
    if (channel) launch.channel = channel;
    if (engine === "firefox" && process.env.SRG_FIREFOX_PROFILE) {
        context = await playwright[engine].launchPersistentContext(process.env.SRG_FIREFOX_PROFILE,
            {...launch, acceptDownloads: true, locale: "ko-KR"});
        browser = context.browser();
    } else {
        browser = await playwright[engine].launch(launch);
        context = await browser.newContext({acceptDownloads: true, locale: "ko-KR", viewport: {width: 1365, height: 1000}});
    }
    metadata.browserVersion = browser.version();
    const page = await context.newPage();
    page.on("pageerror", error => errors.push(String(error)));
    page.on("console", message => { if (message.type() === "error") consoleErrors.push(message.text()); });
    await page.goto(url, {waitUntil: "domcontentloaded", timeout: 60000});
    report = await page.evaluate("(" + source + ")(" + JSON.stringify(config) + ")");
    await page.screenshot({path: path.join(outputDirectory, "final-page.png"), fullPage: true});
    if (errors.length || consoleErrors.length) report.complete = false;
} catch (error) {
    report = {complete: false, failure: String(error), stack: error?.stack};
} finally {
    if (context) await context.close();
    if (browser) await browser.close();
    const result = {metadata, report, pageErrors: errors, consoleErrors};
    await fs.writeFile(path.join(outputDirectory, "desktop-browser-results.json"), JSON.stringify(result, null, 2) + "\n");
    console.log(JSON.stringify({outputDirectory, complete: report?.complete, passed: report?.passed, required: report?.required}));
    if (!report?.complete) process.exitCode = 1;
}
