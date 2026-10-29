import fs from "node:fs/promises";
import assert from "node:assert/strict";
const html=await fs.readFile("revgeo.html","utf8");
const a=html.indexOf("\t\tconst U64_MAX ="),b=html.indexOf("\t\tfunction paddedHex(",a);
if(a<0||b<a)throw Error("Missing generator");
const convert=new Function(html.slice(a,b)+"\nreturn convertNumber")();
let log=[];
async function* spySample(input) {
  log.push("initial:"+input);
  try {
    while(true) {
      const word=yield;
      log.push("supp:"+word.value);
      if(word.value===123n)return [34,128];
    }
  } finally {
    log.push("closed");
  }
}
const generator=convert(0n,spySample);
let step=await generator.next();
assert.equal(step.done,false);
assert.deepEqual(log,["initial:0"]);
step=await generator.next({value:0n,bitsRemaining:64});
assert.equal(step.done,false);
assert.deepEqual(log,["initial:0","supp:0"]);
step=await generator.next({value:123n,bitsRemaining:64});
assert.equal(step.done,false);
assert.deepEqual(log,["initial:0","supp:0","supp:123","closed"]);
step=await generator.next({value:255n,bitsRemaining:64});
assert.equal(step.done,false);
assert.deepEqual(log,["initial:0","supp:0","supp:123","closed"]);
await generator.return();
log=[];
const canceled=convert(0n,spySample);
assert.equal((await canceled.next()).done,false);
await canceled.return();
assert.deepEqual(log,["initial:0","closed"]);
console.log(JSON.stringify({immediateSampling:true,skipOnceComplete:true,cleanupOnCancellation:true}));
