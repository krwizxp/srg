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
const requestFailures = [];
const networkPreparation = {complete: false, externalStylesheets: [], attempts: [], requestFailures: [], consoleErrors: [], pageErrors: []};
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
    const preparationPage = await context.newPage();
    preparationPage.on("requestfailed", request => networkPreparation.requestFailures.push({url: request.url(), error: request.failure()?.errorText, resourceType: request.resourceType(), timing: request.timing()}));
    preparationPage.on("console", message => { if (message.type() === "error") networkPreparation.consoleErrors.push({text: message.text(), location: message.location()}); });
    preparationPage.on("pageerror", error => networkPreparation.pageErrors.push(String(error)));
    await preparationPage.goto(url, {waitUntil: "domcontentloaded", timeout: 60000});
    networkPreparation.externalStylesheets = await preparationPage.evaluate(() => [...document.querySelectorAll('link[rel~="stylesheet"][href]')].map(link => link.href).filter(href => new URL(href).origin !== location.origin));
    if (!networkPreparation.externalStylesheets.length) throw new Error("No production external stylesheet found for strict guest network preparation");
    for (const resourceUrl of networkPreparation.externalStylesheets) {
        const deadline = Date.now() + 90000;
        let ready = false;
        while (!ready && Date.now() < deadline) {
            const attempt = await preparationPage.evaluate(async target => {
                try {
                    const response = await fetch(target, {cache: "reload", signal: AbortSignal.timeout(10000)});
                    const bytes = await response.arrayBuffer();
                    return {url: target, status: response.status, bytes: bytes.byteLength, contentType: response.headers.get("content-type"), ready: response.status === 200 && bytes.byteLength > 0};
                } catch (error) { return {url: target, ready: false, error: String(error)}; }
            }, resourceUrl);
            networkPreparation.attempts.push({...attempt, atEpochMs: Date.now()});
            ready = attempt.ready;
            if (!ready) await new Promise(resolve => setTimeout(resolve, 2000));
        }
        if (!ready) throw new Error("Guest browser could not read the production HTTPS resource within 90 seconds: " + resourceUrl);
    }
    networkPreparation.complete = true;
    await fs.writeFile(path.join(outputDirectory, "android-connectivity-after-preparation.log"), shell(["dumpsys", "connectivity"]) + "\n");
    await context.close();
    context = undefined;
    context = await device.launchBrowser({pkg: metadata.browserPackage, acceptDownloads: true});
    metadata.freshBrowserAfterNetworkPreparation = true;
    const page = await context.newPage();
    page.on("pageerror", error => errors.push(String(error)));
    page.on("console", message => { if (message.type() === "error") consoleErrors.push({text: message.text(), location: message.location()}); });
    page.on("requestfailed", request => requestFailures.push({url: request.url(), error: request.failure()?.errorText, resourceType: request.resourceType(), timing: request.timing()}));
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
    const result = {metadata, report, pageErrors: errors, consoleErrors, requestFailures, networkPreparation};
    await fs.writeFile(path.join(outputDirectory, "android-browser-results.json"), JSON.stringify(result, null, 2) + "\n");
    console.log(JSON.stringify({outputDirectory, complete: report?.complete, passed: report?.passed, required: report?.required}));
    if (!report?.complete) process.exitCode = 1;
}
