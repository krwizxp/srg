import assert from "node:assert/strict";
import fs from "node:fs";

const evidence = "/workspace/scratch/253e85c45315/srg-wasm-evidence";
const product = "/workspace/scratch/253e85c45315/srg";
const paths = process.argv.length > 2
    ? process.argv.slice(2)
    : [evidence + "/candidates/a/source/coordinates.wasm", product + "/coordinates.wasm"];
const reports = [];
for (const path of paths) {
    const { instance } = await WebAssembly.instantiate(fs.readFileSync(path));
    const kernel = instance.exports;
    const pointer = kernel.srg_input_ptr();
    const write = (header, values) => {
        const length = header + values.length * 8;
        const bytes = new Uint8Array(kernel.memory.buffer, pointer, length);
        bytes.fill(0);
        const data = new DataView(kernel.memory.buffer, pointer, length);
        values.forEach((value, index) => data.setFloat64(header + index * 8, value, true));
        return length;
    };
    let parityChecks = 0;
    for (const endpoints of [[], [1], [1, 2], [1, 2, 3], Array.from({ length: 4092 }, (_, index) => index / 100)]) {
        const length = write(32, endpoints);
        for (const base of [0, 1]) {
            for (const latitude of [-1, 0, 1, 1.5, 2, 10.5, 40.91, 100]) {
                const expected = base ^ (endpoints.filter((value) => value <= latitude).length % 2);
                assert.equal(kernel.srg_page_relation(0, endpoints.length, base, latitude, 0, length), expected);
                parityChecks++;
            }
        }
    }
    const rectangle = [0, 0, 2, 0, 2, 2, 0, 2, 0, 0];
    const length = write(40, rectangle);
    const chainResults = [];
    for (const [latitude, longitude, expected] of [[1, 1, 1], [1, 3, 0], [1, 0, 3], [1, 2, 2], [0, 1, 3]]) {
        const result = kernel.srg_page_relation(1, 4, 0, latitude, longitude, length);
        assert.equal(result, expected);
        chainResults.push(result);
    }
    const malformed = [
        [0, 0xffffffff, 0, 0, 0, 0],
        [1, 0xffffffff, 0, 0, 0, 0],
        [0, 4093, 0, 0, 0, 32768],
        [1, 1025, 0, 0, 0, 16456],
        [2, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 32],
        [1, 4, 0, NaN, 0, length],
        [0, 0, 0, 0, 0, 0xffffffff],
    ];
    for (const args of malformed) assert.equal(kernel.srg_page_relation(...args), -1);
    reports.push({ path, parityChecks, chainResults, malformedChecks: malformed.length, memoryBytes: kernel.memory.buffer.byteLength });
}
if (process.argv.length === 2) {
    fs.writeFileSync(evidence + "/rust/b-abi-check.json", JSON.stringify({ passed: true, reports }, null, 2) + "\n");
}
process.stdout.write(JSON.stringify({ passed: true, reports }) + "\n");
