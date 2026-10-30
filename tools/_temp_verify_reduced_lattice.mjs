import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {hrtime} from 'node:process';

const expected = new Map([
    [4, {visited_candidates: 256, accepted_candidates: 137, rejected_candidates: 119, component_counts: [128, 8, 1], accepted_word_sum_mod_2_64: '13858', accepted_word_square_sum_mod_2_64: '1984600', accepted_word_xor: '46'}],
    [8, {visited_candidates: 65536, accepted_candidates: 35072, rejected_candidates: 30464, component_counts: [32768, 2048, 256], accepted_word_sum_mod_2_64: '924558208', accepted_word_square_sum_mod_2_64: '33903180241280', accepted_word_xor: '0'}],
    [16, {visited_candidates: 4294967296, accepted_candidates: 2298478592, rejected_candidates: 1996488704, component_counts: [2147483648, 134217728, 16777216], accepted_word_sum_mod_2_64: '3974413372786475008', accepted_word_square_sum_mod_2_64: '17471634226993430528', accepted_word_xor: '0'}]
]);

function execute(program, bits) {
    const start = hrtime.bigint();
    const result = JSON.parse(execFileSync(program, [String(bits)], {encoding: 'utf8', timeout: 180000}));
    const elapsed = Number(hrtime.bigint() - start) / 1e9;
    assert.equal(result.axis_bits, bits);
    assert.equal(result.word_bits, bits * 2);
    for (const [key, value] of Object.entries(expected.get(bits))) assert.deepEqual(result[key], value, key);
    return {result, elapsed};
}

function median(values) {
    const sorted = values.toSorted((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
}

const rust = process.argv[2];
const c = process.argv[3];
for (const bits of [4, 8, 16]) {
    const a = execute(rust, bits);
    if (c) {
        const b = execute(c, bits);
        const {elapsed_seconds: ignoredRust, ...rustResult} = a.result;
        const {elapsed_seconds: ignoredC, ...cResult} = b.result;
        assert.deepEqual(rustResult, cResult);
        assert.equal(typeof ignoredRust, 'number');
        assert.equal(typeof ignoredC, 'number');
    }
    process.stdout.write(JSON.stringify({bits, native_rust_seconds: a.elapsed, matched: true}) + '\n');
}
if (c) {
    const rustTimes = [];
    const cTimes = [];
    for (let iteration = 0; iteration < 3; iteration++) {
        const order = iteration % 2 === 0 ? [[c, cTimes], [rust, rustTimes]] : [[rust, rustTimes], [c, cTimes]];
        for (const [binary, results] of order) results.push(execute(binary, 16).elapsed);
    }
    const ratio = median(rustTimes) / median(cTimes);
    process.stdout.write(JSON.stringify({c_seconds: cTimes, rust_seconds: rustTimes, median_ratio: ratio, non_regression_limit: 1.2}) + '\n');
    assert.ok(ratio <= 1.2, 'Rust exhaustive enumeration is more than 20 percent slower');
}
