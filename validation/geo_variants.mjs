import {readFileSync,existsSync} from "node:fs";
import {createHash} from "node:crypto";

const originals = new Map([
    ["revgeo.html",readFileSync("revgeo.html")],
    ["korea_boundary.js",readFileSync("korea_boundary.js","utf8")],
    ["srg_parallel.js",readFileSync("srg_parallel.js","utf8")],
    ["srg_worker.js",readFileSync("srg_worker.js","utf8")]
]);
const atlasOriginal=readFileSync("data/korea-boundary-v4/atlas-8.bin");
function replaceOnce(text,original,replacement,name){
    const count=text.split(original).length-1;
    if(count!==1)throw Error(name+" expected once but found "+count);
    return text.replace(original,replacement);
}
function spatial(source){
    source=replaceOnce(source,
        "    const blocks = new Array(values.length);",
        "    const blocks = new Array(values.length);\n    const schedule = values.map((value, index) => ({value, index, selector: Number(value >> 42n)}));\n    schedule.sort((a, b) => a.selector - b.selector);\n    const active = new Map();",
        "spatial schedule");
    source=replaceOnce(source,
        "                worker.postMessage({ start, values: values.slice(start, end) });",
        "                const batch = schedule.slice(start, end);\n                active.set(worker, batch);\n                worker.postMessage({ start, values: batch.map(item => item.value) });",
        "spatial dispatch");
    source=replaceOnce(source,
        "                        for (let i = 0; i < data.blocks.length; i++)\n                            blocks[data.start + i] = data.blocks[i];",
        "                        const batch = active.get(worker);\n                        if (!batch || batch.length !== data.blocks.length || data.start !== nextExpected.get(worker)) {\n                            rejectPending(new Error(\"Web Worker 결과 순서 오류\"));\n                            return;\n                        }\n                        for (let i = 0; i < data.blocks.length; i++)\n                            blocks[batch[i].index] = data.blocks[i];\n                        active.delete(worker);",
        "spatial restore");
    source=replaceOnce(source,
        "    const active = new Map();",
        "    const active = new Map();\n    const nextExpected = new Map();",
        "spatial expectation");
    source=replaceOnce(source,
        "                active.set(worker, batch);",
        "                active.set(worker, batch);\n                nextExpected.set(worker, start);",
        "spatial expectation dispatch");
    return source;
}
function earlyAtlas(source){
    const old="        return {bytes, data, bounds, gridWest, gridSouth, scale, columns, rows, tileIndex,\n            tileCount, partCount, partOffset, pageCount, pagePart, southArea, northArea, polygons, rings, vertices,\n            atlas: null, atlasRequested: false, atlasUnavailable: false, mixedChecks: 0};";
    const replacement="        const boundary = {bytes, data, bounds, gridWest, gridSouth, scale, columns, rows, tileIndex,\n            tileCount, partCount, partOffset, pageCount, pagePart, southArea, northArea, polygons, rings, vertices,\n            atlas: null, atlasRequested: false, atlasUnavailable: false, mixedChecks: 0};\n        requestAtlas(boundary);\n        return boundary;";
    return replaceOnce(source,old,replacement,"early atlas");
}
function pageLatitude(source){
    source=replaceOnce(source,
        "        for (let index = 0; index <= count; index++) {\n            const longitude = coordinates.getFloat64(40 + index * 16, true);",
        "        let minLatitude = Infinity;\n        let maxLatitude = -Infinity;\n        for (let index = 0; index <= count; index++) {\n            const longitude = coordinates.getFloat64(40 + index * 16, true);",
        "ring bbox validation");
    source=replaceOnce(source,
        '            requireBoundary(Number.isFinite(longitude) && Number.isFinite(latitude)\n                && longitude >= west && longitude <= east && latitude >= south && latitude <= north,\n            "대한민국 경계 정점 범위 오류");\n        }\n        return {tile, kind, count, first, hole, ring: fieldA, polygon: fieldB, data: coordinates};',
        '            requireBoundary(Number.isFinite(longitude) && Number.isFinite(latitude)\n                && longitude >= west && longitude <= east && latitude >= south && latitude <= north,\n            "대한민국 경계 정점 범위 오류");\n            minLatitude = Math.min(minLatitude, latitude);\n            maxLatitude = Math.max(maxLatitude, latitude);\n        }\n        return {tile, kind, count, first, hole, ring: fieldA, polygon: fieldB,\n            minLatitude, maxLatitude, data: coordinates};',
        "ring bbox return");
    source=replaceOnce(source,
        "        let start = [page.data.getFloat64(40, true), page.data.getFloat64(48, true)];",
        "        if (latitude < page.minLatitude || latitude > page.maxLatitude) continue;\n        let start = [page.data.getFloat64(40, true), page.data.getFloat64(48, true)];",
        "skip ring latitude");
    return source;
}
function enlargeAtlas(subdivision){
    const refinedPath="validation/geo_atlas"+subdivision+".bin";
    if(existsSync(refinedPath))return new Uint8Array(readFileSync(refinedPath));
    if(subdivision!==16&&subdivision!==32)throw Error("unsupported trial size");
    const view=new DataView(atlasOriginal.buffer,atlasOriginal.byteOffset,atlasOriginal.byteLength);
    const originalSubdivision=view.getUint32(12,true);
    const mixedCount=view.getUint32(16,true);
    if(originalSubdivision!==8||mixedCount<=0)throw Error("original atlas format unexpected");
    const bytes=new Uint8Array(96+mixedCount*subdivision*subdivision/4);
    bytes.set(atlasOriginal.subarray(0,96));
    const header=new DataView(bytes.buffer);
    header.setUint32(12,subdivision,true);
    header.setUint32(88,bytes.length-96,true);
    const factor=subdivision/8;
    for(let ordinal=0;ordinal<mixedCount;ordinal++){
        for(let row=0;row<subdivision;row++){
            for(let col=0;col<subdivision;col++){
                const sourceCell=(ordinal*8+Math.floor(row/factor))*8+Math.floor(col/factor);
                const src=(atlasOriginal[96+(sourceCell>>>2)]>>>((sourceCell&3)*2))&3;
                const dst=(ordinal*subdivision+row)*subdivision+col;
                bytes[96+(dst>>>2)]|=src<<((dst&3)*2);
            }
        }
    }
    return bytes;
}
const memory=new Map();
export function variant(name){
    if(memory.has(name))return memory.get(name);
    const spec=new Set(name.split("+"));
    if(name==="baseline")spec.clear();
    if([...spec].some(x=>!["p1","p2","p3-16","p3-32","p4"].includes(x)))throw Error("Invalid mode "+name);
    let parallel=originals.get("srg_parallel.js");
    let boundary=originals.get("korea_boundary.js");
    let extra=new Map();
    if(spec.has("p1"))parallel=spatial(parallel);
    if(spec.has("p2"))boundary=earlyAtlas(boundary);
    if(spec.has("p4"))boundary=pageLatitude(boundary);
    for(const n of [16,32]){
        if(!spec.has("p3-"+n))continue;
        const bytes=enlargeAtlas(n);
        const hash=createHash("sha256").update(bytes).digest("hex");
        boundary=replaceOnce(boundary,'const ATLAS_SHA256 = "a55dcf2e37a4500d6c30de3f5dad42131758415bcfafac79e88d76fb88b0e84d";',
            'const ATLAS_SHA256 = "'+hash+'";',"atlas sha");
        boundary=replaceOnce(boundary,'subdivision === 8','subdivision === '+n,"atlas subdivision");
        boundary=replaceOnce(boundary,'download("atlas-8.bin", 41328, ATLAS_SHA256)','download("atlas-'+n+'.bin", '+bytes.length+', ATLAS_SHA256)',"atlas request");
        extra.set("data/korea-boundary-v4/atlas-"+n+".bin",Buffer.from(bytes));
    }
    const result={parallel,boundary,extras:extra};
    memory.set(name,result);
    return result;
}
export function original(name){return originals.get(name);}
export function modes(){return ["baseline","p1","p2","p4","p3-16","p3-32","p1+p2","p1+p2+p4"];}
