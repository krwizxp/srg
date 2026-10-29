async function srgPlatformContract(config) {
    const cases = [];
    const errors = [];
    const assert = (condition, message) => { if (!condition) throw new Error(message); };
    const byId = id => {
        const element = document.getElementById(id);
        assert(element !== null, "Missing element: " + id);
        return element;
    };
    const wait = async (condition, label, timeout = 30000) => {
        const started = performance.now();
        while (!condition()) {
            if (performance.now() - started >= timeout) throw new Error("Timed out: " + label);
            await new Promise(resolve => setTimeout(resolve, 25));
        }
    };
    const check = async (name, operation) => {
        try {
            const detail = await operation();
            cases.push({name, status: "passed", detail: detail ?? null});
        } catch (error) {
            cases.push({name, status: "failed", error: String(error), stack: error?.stack ?? null});
            throw error;
        }
    };
    const captureError = event => errors.push(String(event.error ?? event.reason ?? event.message));
    addEventListener("error", captureError);
    addEventListener("unhandledrejection", captureError);
    const bits = value => {
        const view = new DataView(new ArrayBuffer(8));
        view.setFloat64(0, value, true);
        return view.getBigUint64(0, true).toString();
    };
    const setValue = (id, value) => {
        const element = byId(id);
        element.value = value;
        element.dispatchEvent(new Event("input", {bubbles: true}));
    };
    const rows = config.fixtures.rows;
    let api;
    let failure;
    try {
        await check("required_features_and_nonisolated_origin", async () => {
            const features = {
                secureContext: isSecureContext,
                webAssembly: typeof WebAssembly?.instantiate === "function",
                bigint: typeof BigInt === "function",
                dataViewU64: typeof DataView.prototype.getBigUint64 === "function",
                cryptoDigest: typeof crypto?.subtle?.digest === "function",
                cryptoEntropy: typeof crypto?.getRandomValues === "function",
                abortAny: typeof AbortSignal.any === "function",
                abortTimeout: typeof AbortSignal.timeout === "function",
                promiseResolvers: typeof Promise.withResolvers === "function",
                promiseTry: typeof Promise.try === "function",
                iteratorMap: typeof [][Symbol.iterator]().map === "function",
                toHex: typeof Uint8Array.prototype.toHex === "function",
                popover: typeof HTMLElement.prototype.hidePopover === "function",
                invokerCommands: "commandForElement" in HTMLButtonElement.prototype,
                fileStream: typeof File.prototype.stream === "function",
                fetch: typeof fetch === "function",
                textDecoder: typeof TextDecoder === "function"
            };
            assert(Object.values(features).every(Boolean), "Missing features: " + JSON.stringify(features));
            assert(location.protocol === "http:" || location.protocol === "https:", "Wrong origin protocol");
            assert(!crossOriginIsolated, "Probe must prove operation without cross-origin isolation");
            const entropy = new BigUint64Array(1);
            crypto.getRandomValues(entropy);
            assert(rows.length === 50, "Expected fixed 50-row corpus");
            return {features, origin: location.origin, crossOriginIsolated,
                userAgent: navigator.userAgent, platform: navigator.platform,
                hardwareConcurrency: navigator.hardwareConcurrency,
                decompressionStreamRequired: false,
                decompressionStreamAvailable: typeof DecompressionStream === "function",
                pointer: matchMedia("(pointer: coarse)").matches ? "coarse" : "fine",
                viewport: [innerWidth, innerHeight]};
        });
        if (config.wasm) {
            await check("actual_wasm_artifact_and_unshared_memory", async () => {
                const response = await fetch(new URL(config.wasmPath, location.href));
                assert(response.ok, "WASM HTTP status " + response.status);
                const bytes = await response.arrayBuffer();
                assert(bytes.byteLength <= 262144, "WASM artifact exceeds byte budget");
                const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)).toHex();
                assert(digest === config.wasmSha256, "WASM artifact identity mismatch");
                const module = await WebAssembly.compile(bytes);
                const imports = WebAssembly.Module.imports(module);
                assert(imports.length === 0, "Unexpected host imports: " + JSON.stringify(imports));
                const instance = await WebAssembly.instantiate(module);
                assert(instance.exports.srg_abi_version() === 1, "Unexpected ABI version");
                instance.exports.srg_input_ptr();
                instance.exports.srg_output_ptr();
                const memory = instance.exports.memory;
                assert(memory instanceof WebAssembly.Memory, "Missing memory");
                assert(memory.buffer instanceof ArrayBuffer, "Shared memory must not be required");
                assert(memory.buffer.byteLength <= 16 * 1024 * 1024, "WASM initialized memory exceeds budget");
                return {sha256: digest, bytes: bytes.byteLength, imports,
                    exports: WebAssembly.Module.exports(module),
                    initializedRawMemoryBytes: memory.buffer.byteLength,
                    contentType: response.headers.get("content-type"),
                    streamingMimeRequired: false, sharedMemory: false};
            });
        }
        await check("coordinate_batch50_exact_bits_and_supplements", async () => {
            const boundary = await import(new URL(config.boundaryPath, location.href).href);
            api = config.wasm ? await boundary.loadCoordinates() : boundary;
            const comparisons = [];
            for (const row of rows) {
                const generator = api.sampleKorea(BigInt(row.num64));
                let step = await generator.next();
                let used = 0;
                while (!step.done) {
                    assert(used < row.koreaSupplements.length, "Unexpected extra supplement at " + row.num64);
                    step = await generator.next({value: BigInt(row.koreaSupplements[used++]), bitsRemaining: 64});
                }
                assert(used === row.koreaSupplementCount, "Supplement count mismatch at " + row.num64);
                const koreaBits = step.value.map(bits);
                assert(JSON.stringify(koreaBits) === JSON.stringify(row.koreaBits), "Korea bits mismatch at " + row.num64);
                let worldBits = null;
                if (config.wasm) {
                    worldBits = api.worldCoordinates(BigInt(row.num64)).map(bits);
                    assert(JSON.stringify(worldBits) === JSON.stringify(row.worldBits), "World bits mismatch at " + row.num64);
                }
                comparisons.push({num64: row.num64, koreaBits, worldBits, supplements: used});
            }
            return {count: comparisons.length, comparisons};
        });
        const rejected = rows.find(row => row.koreaSupplementCount > 0);
        assert(rejected, "Corpus must include a rejected first candidate");
        await check("supplement_full_word_and_input_error_contracts", async () => {
            const diagnostics = [];
            for (const invalid of [{value: 0n, bitsRemaining: 63}, {value: -1n, bitsRemaining: 64},
                {value: 18446744073709551616n, bitsRemaining: 64}]) {
                const generator = api.sampleKorea(BigInt(rejected.num64));
                const first = await generator.next();
                assert(!first.done, "Expected rejected first candidate");
                let caught;
                try { await generator.next(invalid); } catch (error) { caught = String(error); }
                assert(caught?.includes("전체 64비트"), "Missing full-word error: " + caught);
                diagnostics.push(caught);
            }
            for (const invalid of [-1n, 18446744073709551616n]) {
                let caught;
                try { await api.sampleKorea(invalid).next(); } catch (error) { caught = String(error); }
                assert(caught?.includes("범위"), "Missing range error: " + caught);
                diagnostics.push(caught);
            }
            return {rejectedCases: diagnostics.length, diagnostics};
        });
        await check("supplement1024_last_acceptance_and_overflow", async () => {
            const acceptWord = BigInt(rejected.koreaSupplements.at(-1));
            const rejectWord = BigInt(rejected.num64);
            const generator = api.sampleKorea(rejectWord);
            assert(!(await generator.next()).done, "Expected rejected first candidate");
            for (let i = 0; i < 1023; i++) {
                assert(!(await generator.next({value: rejectWord, bitsRemaining: 64})).done, "Premature completion");
            }
            const final = await generator.next({value: acceptWord, bitsRemaining: 64});
            assert(final.done, "1024th supplemental candidate was not accepted");
            assert(JSON.stringify(final.value.map(bits)) === JSON.stringify(rejected.koreaBits), "Last candidate bits differ");
            const overflow = api.sampleKorea(rejectWord);
            assert(!(await overflow.next()).done, "Expected rejected first candidate");
            for (let i = 0; i < 1023; i++) await overflow.next({value: rejectWord, bitsRemaining: 64});
            let caught;
            try { await overflow.next({value: rejectWord, bitsRemaining: 64}); } catch (error) { caught = String(error); }
            assert(caught?.includes("시도 횟수 초과"), "Missing supplement exhaustion error: " + caught);
            return {lastAccepted: 1024, overflowRejected: 1024, diagnostic: caught};
        });
        const row = rows[0];
        let previousOutput;
        await check("manual_ui_fixed_conversion", async () => {
            byId("manual-tab").click();
            setValue("num64", row.num64);
            setValue("suppValues", row.supplements.join("\n"));
            byId("auto-supp").checked = false;
            byId("runBtn").click();
            await wait(() => byId("meta").textContent.includes("완료") && !byId("runBtn").disabled, "manual conversion", 90000);
            previousOutput = byId("output").textContent;
            const koreaLine = previousOutput.match(/대한민국 위경도:\s*([^,\n]+),\s*([^\n]+)/);
            const worldLine = previousOutput.match(/세계 위경도:\s*([^,\n]+),\s*([^\n]+)/);
            assert(koreaLine && worldLine, "Missing coordinate output");
            const koreaBits = koreaLine.slice(1).map(Number).map(bits);
            const worldBits = worldLine.slice(1).map(Number).map(bits);
            assert(JSON.stringify(koreaBits) === JSON.stringify(row.koreaBits), "UI Korea mismatch");
            assert(JSON.stringify(worldBits) === JSON.stringify(row.worldBits), "UI world mismatch");
            const meta = byId("meta").textContent;
            if (row.fullSupplementCount) assert(meta.includes("보조 " + row.fullSupplementCount + "개"), "UI consumption mismatch: " + meta);
            return {num64: row.num64, koreaBits, worldBits, meta,
                interaction: "Real browser DOM click/input event paths; not physical touch-gesture coverage"};
        });
        await check("manual_supplement_cancel_preserves_prior_output", async () => {
            setValue("num64", "0");
            setValue("suppValues", "");
            byId("auto-supp").checked = false;
            byId("runBtn").click();
            await wait(() => byId("supp-dialog").open, "supplement dialog", 60000);
            byId("supp-dismiss").click();
            await wait(() => !byId("supp-dialog").open && !byId("runBtn").disabled, "cancel completion");
            assert(byId("output").textContent === previousOutput, "Cancelled work changed previous output");
            await new Promise(resolve => setTimeout(resolve, 250));
            assert(byId("output").textContent === previousOutput, "Late cancelled work changed output");
            return {priorOutputPreserved: true, dialogClosed: true};
        });
        await check("invalid_manual_input_reports_error", async () => {
            setValue("num64", "not-an-integer");
            byId("runBtn").click();
            await wait(() => byId("num64-error").textContent.trim().length > 0, "input diagnostic");
            assert(byId("output").textContent === previousOutput, "Invalid input changed previous output");
            return {diagnostic: byId("num64-error").textContent};
        });
        await check("converted_output_file_parse_and_clear", async () => {
            byId("useOutput").click();
            await wait(() => byId("coords-count").textContent.trim() === "2", "output file parse");
            byId("manual-tab").click();
            byId("clearBtn").click();
            await wait(() => byId("clear-dialog").open, "clear confirmation");
            byId("clear-confirm").click();
            await wait(() => byId("output").querySelector(".code-content").textContent === "", "clear result");
            await new Promise(resolve => setTimeout(resolve, 250));
            assert(byId("output").querySelector(".code-content").textContent === "", "Late work repopulated cleared output");
            return {parsedCoordinates: 2, clearedWithoutStaleOutput: true};
        });
        await check("no_asynchronous_errors", async () => {
            assert(errors.length === 0, "Asynchronous errors: " + JSON.stringify(errors));
            return {errors};
        });
    } catch (error) {
        failure = String(error);
    } finally {
        removeEventListener("error", captureError);
        removeEventListener("unhandledrejection", captureError);
    }
    const required = config.wasm ? 10 : 9;
    const passed = cases.filter(test => test.status === "passed").length;
    return {required, passed, failed: cases.filter(test => test.status === "failed").length,
        complete: passed === required && !failure, failure: failure ?? null, cases,
        fixtureBaseline: config.fixtures.metadata ?? config.fixtures.baseline ?? null};
}
