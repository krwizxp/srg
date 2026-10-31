import {readFileSync,createReadStream,statSync} from "node:fs";
import {createHash} from "node:crypto";
import {createServer} from "node:http";
import {resolve,sep} from "node:path";
import {chromium,firefox,webkit} from "playwright";
import {variant,modes,original} from "./geo_variants.mjs";

const root=process.cwd();
const engineName=process.env.GEO_BROWSER||"chromium";
const engine={chromium,firefox,webkit}[engineName];
if(!engine)throw Error("Unknown browser "+engineName);
const rustExpected=readFileSync("validation/geo_rust_full.txt","utf8").trim().split(/\r?\n/);
if(rustExpected.length!==10000)throw Error("Rust baseline rows missing");
function replaceOne(source,original,replacement,label){
    const n=source.split(original).length-1;
    if(n!==1)throw Error("Parity overlay "+label+" appears "+n+" times");
    return source.replace(original,replacement);
}
function paritySources(mode){
    const source=variant(mode);
    let parallel=source.parallel;
    let worker=original("srg_worker.js");
    parallel=replaceOne(parallel,"    const blocks = new Array(values.length);",
        "    const blocks = new Array(values.length);\n    const reads = new Array(values.length);","read output");
    parallel=replaceOne(parallel,"resolve({ blocks, supplements: supplementalReads });",
        "resolve({ blocks, supplements: supplementalReads, reads });","resolve");
    const spatial=mode.split("+").includes("p1");
    const store=spatial?"blocks[batch[i].index] = data.blocks[i];":"blocks[data.start + i] = data.blocks[i];";
    const update=spatial?"reads[batch[i].index] = data.reads[i];":"reads[data.start + i] = data.reads[i];";
    parallel=replaceOne(parallel,store,store+"\n                            "+update,"read index");
    worker=replaceOne(worker,"    const blocks = new Array(values.length);",
        "    const blocks = new Array(values.length);\n    const reads = new Array(values.length);","worker reads");
    worker=replaceOne(worker,"        const value = values[i];",
        "        const value = values[i];\n        const supplementState = {value: value ^ 0xa5a5a5a5a5a5a5a5n};\n        offset = buffer.length;","per input seed");
    worker=replaceOne(worker,"                crypto.getRandomValues(buffer);",
        "                for (let index = 0; index < buffer.length; index++) buffer[index] = splitmix64(supplementState);","deterministic bytes");
    worker=replaceOne(worker,"        blocks[i] = formatOutput(step.value);",
        "        blocks[i] = formatOutput(step.value);\n        reads[i] = recordSupplements;","worker read count");
    worker=replaceOne(worker,"return { start, blocks, supplements };",
        "return { start, blocks, supplements, reads };","worker return");
    worker += "\nfunction splitmix64(state) {\n"
        + "    const mask = (1n << 64n) - 1n;\n"
        + "    state.value = (state.value + 0x9e3779b97f4a7c15n) & mask;\n"
        + "    let x = state.value;\n"
        + "    x = ((x ^ (x >> 30n)) * 0xbf58476d1ce4e5b9n) & mask;\n"
        + "    x = ((x ^ (x >> 27n)) * 0x94d049bb133111ebn) & mask;\n"
        + "    return (x ^ (x >> 31n)) & mask;\n"
        + "}\n";
    return {parallel,worker};
}
const parityCache=new Map();
const network=new Map();
const server=createServer((req,res)=>{
    const pathname=new URL(req.url,"http://127.0.0.1").pathname;
    const [mode,phase,...parts]=pathname.split("/").slice(1);
    if(!mode||!["prod","parity"].includes(phase)) {res.writeHead(404);res.end();return;}
    let sources;
    try{sources=variant(mode)}catch(error){res.writeHead(400);res.end(String(error));return;}
    const file=parts.join("/")||"revgeo.html";
    let bytes=null;
    if(file==="srg_parallel.js")bytes=phase==="parity"?(parityCache.get(mode)??parityCache.set(mode,paritySources(mode)).get(mode)).parallel:sources.parallel;
    else if(file==="srg_worker.js")bytes=phase==="parity"?(parityCache.get(mode)??parityCache.set(mode,paritySources(mode)).get(mode)).worker:original("srg_worker.js");
    else if(file==="korea_boundary.js")bytes=sources.boundary;
    else if(sources.extras.has(file))bytes=sources.extras.get(file);
    if(bytes===null) {
        const actual=resolve(root,file);
        if(!actual.startsWith(root+sep)){res.writeHead(403);res.end();return;}
        try{
            const stat=statSync(actual);
            if(!stat.isFile())throw Error("Not file");
            res.writeHead(200,{"Content-Length":stat.size,"Cache-Control":file.endsWith(".bin")?"public,max-age=3600":"no-store","Content-Type":file.endsWith(".js")?"text/javascript":file.endsWith(".html")?"text/html;charset=utf-8":"application/octet-stream"});
            if(file.endsWith(".bin")){
                const key=mode+"/"+phase;
                const count=network.get(key)||{requestCount:0,partRequests:0,atlasRequests:0,bytes:0};
                count.requestCount++;
                count.partRequests+=file.includes("part-")?1:0;
                count.atlasRequests+=file.includes("atlas-")?1:0;
                count.bytes+=stat.size;
                network.set(key,count);
            }
            createReadStream(actual).pipe(res);
        }catch{res.writeHead(404);res.end();}
        return;
    }
    const blob=typeof bytes==="string"?Buffer.from(bytes):bytes;
    res.writeHead(200,{"Content-Type":file.endsWith(".js")?"text/javascript":"application/octet-stream","Content-Length":blob.length,"Cache-Control":file.endsWith(".bin")?"public,max-age=3600":"no-store"});
    if(file.endsWith(".bin")){
        const key=mode+"/"+phase;
        const count=network.get(key)||{requestCount:0,partRequests:0,atlasRequests:0,bytes:0};
        count.requestCount++;
        count.atlasRequests+=file.includes("atlas-")?1:0;
        count.bytes+=blob.length;
        network.set(key,count);
    }
    res.end(blob);
});
await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
const origin="http://127.0.0.1:"+server.address().port;
const browser=await engine.launch({headless:true});
function splitmix64(state){
    const mask=(1n<<64n)-1n;
    state.value=(state.value+0x9e3779b97f4a7c15n)&mask;
    let x=state.value;
    x=((x^(x>>30n))*0xbf58476d1ce4e5b9n)&mask;
    x=((x^(x>>27n))*0x94d049bb133111ebn)&mask;
    return (x^(x>>31n))&mask;
}
const words=[];
const seed={value:0x123456789abcdef0n};
for(let i=0;i<10000;i++)words.push(splitmix64(seed));
function summarize(a){
    const values=[...a].sort((x,y)=>x-y);
    return {medianMs:values[Math.floor(values.length/2)],minMs:values[0],maxMs:values.at(-1),samplesMs:a};
}
function verifyRust(results,n,mode){
    const view=new DataView(new ArrayBuffer(8));
    const bits=value=>{view.setFloat64(0,value,false);return view.getBigUint64(0,false).toString(16).padStart(16,"0");};
    let errors=0;
    let examples=[];
    for(let i=0;i<n;i++){
        const line=results.blocks[i];
        const korea=/대한민국 위경도: ([^,]+), ([^\n]+)/.exec(line);
        const world=/세계 위경도: ([^,]+), ([^\n]+)/.exec(line);
        if(!korea||!world)throw Error("Formatted coordinates missing for "+mode+" at "+i);
        const actual=[bits(Number(korea[1])),bits(Number(korea[2])),bits(Number(world[1])),bits(Number(world[2])),String(results.reads[i])].join(" ");
        if(actual!==rustExpected[i]){
            if(examples.length<3)examples.push({index:i,actual,expected:rustExpected[i]});
            errors++;
        }
    }
    if(errors)throw Error(mode+" Rust parity mismatches "+errors+" "+JSON.stringify(examples));
    return {compared:n,mismatches:0};
}
const final=[];
try{
    for(const n of [1000,10000]){
        const workload=words.slice(0,n).map(String);
        const order=n===1000?["baseline","p1","p2","p4","p3-16","p3-32","p1+p2+p4"]:modes();
        let baselineHash=null,baselineReads=null;
        for(const mode of order){
            let parity,production,issues=[];
            for(const phase of ["prod","parity"]){
                const context=await browser.newContext();
                try{
                    const page=await context.newPage();
                    page.setDefaultTimeout(300000);
                    page.on("pageerror",error=>issues.push(error.message));
                    await page.goto(origin+"/"+mode+"/"+phase+"/revgeo.html",{waitUntil:"domcontentloaded"});
                    const result=await page.evaluate(async ({words,phase})=>{
                        const {convertParallel}=await import("./srg_parallel.js?v=1");
                        const input=words.map(BigInt);
                        const repeat=phase==="prod"?6:1;
                        const times=[],checks=[];
                        let rows;
                        for(let j=0;j<repeat;j++){
                            const start=performance.now();
                            const result=await convertParallel(input,{workers:4});
                            times.push(performance.now()-start);
                            if(result.blocks.length!==input.length)throw Error("Mismatch output count");
                            if(!result.blocks.every((block,i)=>block.includes("64비트 난수: "+input[i].toString())))throw Error("Output order changed");
                            if(phase==="parity")rows={blocks:result.blocks,reads:result.reads,supplements:result.supplements};
                            checks.push({sup:result.supplements,rows:result.blocks.length});
                        }
                        return {times,checks,rows};
                    },{words:workload,phase});
                    if(phase==="parity")parity=result;
                    else production=result;
                }finally{await context.close();}
            }
            if(issues.length)throw Error(engineName+" "+mode+" page errors "+JSON.stringify(issues));
            const check=verifyRust(parity.rows,n,mode);
            const digest=createHash("sha256").update(parity.rows.blocks.join("\n")+"\n").digest("hex");
            const readHash=createHash("sha256").update(parity.rows.reads.join(",")).digest("hex");
            if(mode==="baseline"){baselineHash=digest;baselineReads=readHash;}
            if(baselineHash!==digest||baselineReads!==readHash)throw Error("Variant changed full deterministic output: "+mode);
            const item={engine:engineName,version:browser.version(),mode,n,originalOutputHashMatch:true,rustParity:check,
                ...summarize(production.times.slice(1)),coldMs:production.times[0],substituteWords:parity.rows.supplements,
                networkProduction:network.get(mode+"/prod")||{},networkParity:network.get(mode+"/parity")||{}};
            final.push(item);
            console.log("GEO_VARIANT "+JSON.stringify(item));
        }
    }
    console.log("GEO_RESULTS "+JSON.stringify({engine:engineName,version:browser.version(),runs:final.length,candidates:final.map(x=>({mode:x.mode,n:x.n,medianMs:x.medianMs,bytes:x.networkProduction.bytes}))}));
}finally{
    await browser.close();
    await new Promise(resolve=>server.close(resolve));
}
