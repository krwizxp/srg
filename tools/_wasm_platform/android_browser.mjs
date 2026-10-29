import fs from "node:fs/promises";
import path from "node:path";
import {pathToFileURL} from "node:url";
import {createRequire} from "node:module";
import {execFileSync} from "node:child_process";

const [url, fixturesPath, outputDirectory, wasmSha256, boundaryPath = "./korea_boundary.js?v=6"] = process.argv.slice(2);
if (!url || !fixturesPath || !outputDirectory) throw new Error("Usage: android_browser.mjs URL FIXTURES OUTPUT [WASM_SHA256] [BOUNDARY_PATH]");
await fs.mkdir(outputDirectory, {recursive: true});
const require = createRequire(import.meta.url);
const playwrightPath = process.env.SRG_PLAYWRIGHT_MODULE || require.resolve("playwright");
const {_android} = await import(pathToFileURL(playwrightPath).href);
const source = await fs.readFile(new URL("./browser_contract.js", import.meta.url), "utf8");
const fixtures = JSON.parse(await fs.readFile(fixturesPath, "utf8"));
const config = {fixtures, wasm: Boolean(wasmSha256), wasmSha256,
    wasmPath: "./coordinates.wasm", boundaryPath};
const adb = process.env.SRG_ADB || "adb";
const devices = await _android.devices();
if (devices.length !== 1) throw new Error("Expected exactly one task-owned Android emulator, found " + devices.length);
const device = devices[0];
let context;
let report;
const errors = [];
const consoleErrors = [];
const serial = device.serial();
const shell = args => execFileSync(adb, ["-s", serial, "shell", ...args], {encoding: "utf8"}).trim();
const metadata = {environment: "Android SDK AVD emulator; actual Android browser engine; not device emulation",
    physicalDevice: false, model: device.model(), serial,
    androidRelease: shell(["getprop", "ro.build.version.release"]),
    androidApi: shell(["getprop", "ro.build.version.sdk"]),
    androidAbi: shell(["getprop", "ro.product.cpu.abi"]),
    buildFingerprint: shell(["getprop", "ro.build.fingerprint"]),
    emulator: shell(["getprop", "ro.kernel.qemu"]),
    browserPackage: process.env.SRG_ANDROID_BROWSER_PACKAGE || "com.android.chrome"};
try {
    context = await device.launchBrowser({pkg: metadata.browserPackage, acceptDownloads: true});
    const page = await context.newPage();
    page.on("pageerror", error => errors.push(String(error)));
    page.on("console", message => { if (message.type() === "error") consoleErrors.push(message.text()); });
    await page.goto(url, {waitUntil: "domcontentloaded", timeout: 60000});
    metadata.browserVersion = context.browser()?.version() ?? await page.evaluate(() => navigator.userAgent);
    report = await page.evaluate("(" + source + ")(" + JSON.stringify(config) + ")");
    await page.screenshot({path: path.join(outputDirectory, "final-page.png"), fullPage: true});
    if (errors.length || consoleErrors.length) report.complete = false;
} catch (error) {
    report = {complete: false, failure: String(error), stack: error?.stack};
} finally {
    if (context) await context.close();
    await device.close();
    const result = {metadata, report, pageErrors: errors, consoleErrors};
    await fs.writeFile(path.join(outputDirectory, "android-browser-results.json"), JSON.stringify(result, null, 2) + "\n");
    console.log(JSON.stringify({outputDirectory, complete: report?.complete, passed: report?.passed, required: report?.required}));
    if (!report?.complete) process.exitCode = 1;
}
