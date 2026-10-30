import {readFileSync} from "node:fs";
import {spawnSync, execFileSync} from "node:child_process";
import {performance} from "node:perf_hooks";
const bounds = [124.60971767856728, 33.11235575873201, 131.87278314710602, 38.61370930167485];
const newSource = readFileSync("korea_boundary.js", "utf8");
const oldSource = execFileSync("git", ["show", "ee021b543e8e9cd955e8e23878afba0221d53e33:korea_boundary.js"], {encoding: "utf8"});
function extract(src) {
    const first = src.indexOf("const DEGREES =");
    const last = src.indexOf("const MAX_U64 =");
    if (first < 0 || last <= first) throw Error("sampler source missing");
    return Function(src.slice(first, last) + "\nreturn {candidate, cumulativeArea};")();
}
const old = extract(oldSource);
const now = extract(newSource);
const output = "validation/korea-probe" + (process.platform === "win32" ? ".exe" : "");
const compile = spawnSync("rustc", ["+1.99.1", "--edition=2024", "-O", "validation/geo_candidate_verify.rs", "-o", output], {encoding:"utf8"});
if (compile.status !== 0) throw Error("Rust compile: " + compile.stderr);
const execution = spawnSync(output.startsWith("/") ? output : "./" + output, [], {encoding:"utf8", maxBuffer: 8*1024*1024});
if (execution.status !== 0) throw Error("Rust run: " + execution.stderr);
const expected = execution.stdout.trim().split(/\r?\n/);
const mask = (1n << 64n) - 1n;
let seed = 0x123456789abcdef0n;
function word() {
    seed = (seed + 0x9e3779b97f4a7c15n) & mask;
    let x = seed;
    x = ((x ^ (x >> 30n)) * 0xbf58476d1ce4e5b9n) & mask;
    x = ((x ^ (x >> 27n)) * 0x94d049bb133111ebn) & mask;
    return (x ^ (x >> 31n)) & mask;
}
const words = Array.from({length:25000}, word);
words.push(0n, mask, 0xffffffffn, 0xffffffff00000000n, 0x8000000080000000n);
const bits = new DataView(new ArrayBuffer(8));
function f64Hex(n) {
    bits.setFloat64(0,n,false);
    return bits.getBigUint64(0,false).toString(16).padStart(16,"0");
}
let mismatches=0,maxDifference=0,olderDifference=0,outside=0;
for (let i=0;i<words.length;i++) {
    const w=words[i];
    const high=Number(w >> 32n), low=Number(w & 0xffffffffn);
    const [latitude,longitude]=now.candidate([high,low],bounds);
    const [oldLatitude,oldLongitude]=old.candidate(w,bounds);
    const actual=f64Hex(latitude)+" "+f64Hex(longitude);
    if (expected[i] !== actual) {
        mismatches++;
        if (mismatches<=4) console.log("mismatch",i,w.toString(16),expected[i],actual);
    }
    maxDifference=Math.max(maxDifference,Math.abs(latitude-oldLatitude));
    olderDifference=Math.max(olderDifference,Math.abs(longitude-oldLongitude));
    if (!(latitude>bounds[1] && latitude<bounds[3] && longitude>bounds[0] && longitude<bounds[2])) outside++;
}
const bins=Array(32).fill(0);
const southArea=old.cumulativeArea(bounds[1]),northArea=old.cumulativeArea(bounds[3]);
let maxCdfError=0;
const N=250000;
for(let i=0;i<N;i++){
    const w=word();
    const vertical=(Number(w >> 32n)+0.5)/4294967296;
    const latitude=now.candidate([Number(w >> 32n),Number(w & 0xffffffffn)],bounds)[0];
    const u=(old.cumulativeArea(latitude)-southArea)/(northArea-southArea);
    maxCdfError=Math.max(maxCdfError, Math.abs(u-vertical));
    bins[Math.max(0,Math.min(31,Math.floor(u*32)))]++;
}
const expectedBin=N/32, sigma=Math.sqrt(N*(1/32)*(31/32));
const maxDeviation=Math.max(...bins.map(v=>Math.abs(v-expectedBin)/sigma));
const sampled=words.slice(0,20000);
function time(fn) {
    let sink=0, timings=[];
    for(let warm=0;warm<2;warm++)for(const w of sampled)sink+=fn(w);
    for(let repeat=0;repeat<9;repeat++){
        const start=performance.now();
        for(const w of sampled)sink+=fn(w);
        timings.push((performance.now()-start)/sampled.length*1000);
    }
    timings.sort((a,b)=>a-b);
    return {median_us:timings[4],p95_us:timings[8],sink};
}
const baseline=time(w=>old.candidate(w,bounds)[0]);
const optimized=time(w=>now.candidate([Number(w >> 32n),Number(w&0xffffffffn)],bounds)[0]);
const ratio=optimized.median_us/baseline.median_us;
console.log(JSON.stringify({os:process.platform,arch:process.arch,nativeRust:words.length,bitMismatches:mismatches,maxLatitudeDifferenceDegrees:maxDifference,maxLongitudeDifferenceDegrees:olderDifference,outside,maxCdfError,maxZScore:maxDeviation,baseline,optimized,ratio},null,2));
if (expected.length!==words.length || mismatches || outside || maxCdfError>1e-11 || maxDeviation>6 || ratio>0.60) process.exitCode=1;
