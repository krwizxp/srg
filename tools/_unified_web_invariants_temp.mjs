import fs from "node:fs/promises";
import assert from "node:assert/strict";
const html = await fs.readFile("revgeo.html", "utf8");
const start = html.indexOf("const U64_MAX =");
const stop = html.indexOf("function paddedHex(", start);
assert.ok(start >= 0 && stop > start);
const {convertNumber, createDataSet, fillWord, isComplete} =
    new Function(html.slice(start, stop) +
    "\nreturn {convertNumber,createDataSet,fillWord,isComplete}")();
const bits = (1n << 64n) - 1n;
const record = createDataSet(0n);
fillWord(0n, record, true);
assert.deepEqual(record.hangul, ["가","가","가","가"]);
assert.equal(record.planet, 1);
assert.equal(record.system, 1);
fillWord(bits, record, false);
assert.deepEqual(record.hangul, ["가","가","가","가"]);
assert.equal(record.planet, 1);
assert.equal(record.system, 1);
const actions = [];
async function* fakeKorea(value) {
    actions.push("initial:" + value);
    try {
        while (true) {
            const supp = yield;
            actions.push("supp:" + supp.value);
            if (supp.value === 123n) return [37, 127];
        }
    } finally {
        actions.push("closed");
    }
}
const first = convertNumber(0n, fakeKorea);
assert.equal((await first.next()).done, false);
assert.deepEqual(actions, ["initial:0"]);
assert.equal((await first.next({value:123n,bitsRemaining:64})).done, false);
assert.deepEqual(actions, ["initial:0","supp:123","closed"]);
assert.equal((await first.next({value:255n,bitsRemaining:64})).done, false);
assert.deepEqual(actions, ["initial:0","supp:123","closed"]);
await first.return();
actions.length=0;
const cancelled=convertNumber(0n,fakeKorea);
await cancelled.next();
await cancelled.return();
assert.deepEqual(actions,["initial:0","closed"]);
assert.equal(isComplete(record, {done:false}), false);
console.log(JSON.stringify({sameSource:true,completedFieldsFrozen:true,noKoreaSpecialLoop:true,cancelledCleanup:true}));
