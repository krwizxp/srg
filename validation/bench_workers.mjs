import {readFileSync,createReadStream,statSync} from "node:fs";
import {createServer} from "node:http";
import {resolve,sep} from "node:path";
import {chromium,firefox,webkit} from "playwright";

const expected=readFileSync("validation/bench10000_rust_full.txt","utf8").trim().split(/\r?\n/);
if(expected.length!==10000)throw Error("Rust results missing");
const root=process.cwd();

function replaceOne(source, original, replacement) {
    if(source.split(original).length!==2)throw Error("Test-only worker hook no longer matches production source");
    return source.replace(original,replacement);
}
const productionParallel=readFileSync("srg_parallel.js","utf8");
const productionWorker=readFileSync("srg_worker.js","utf8");
let testParallel=replaceOne(productionParallel,
    "workers = 4, onProgress, signal",
    "workers = 4, deterministic = false, onProgress, signal");
testParallel=replaceOne(testParallel,
    "const blocks = new Array(values.length);",
    "const blocks = new Array(values.length);\n    const reads = new Array(values.length);");
testParallel=replaceOne(testParallel,
    "resolve({ blocks, supplements: supplementalReads });",
    "resolve({ blocks, supplements: supplementalReads, reads });");
testParallel=replaceOne(testParallel,
    "worker.postMessage({ start, values: values.slice(start, end) });",
    "worker.postMessage({ start, values: values.slice(start, end), deterministic });");
testParallel=replaceOne(testParallel,
    "for (let i = 0; i < data.blocks.length; i++)\n                            blocks[data.start + i] = data.blocks[i];",
    "for (let i = 0; i < data.blocks.length; i++) {\n                            blocks[data.start + i] = data.blocks[i];\n                            reads[data.start + i] = data.reads[i];\n                        }");
let testWorker=replaceOne(productionWorker,
    "async function convertBatch({ start, values }) {",
    "async function convertBatch({ start, values, deterministic }) {");
testWorker=replaceOne(testWorker,
    "const blocks = new Array(values.length);",
    "const blocks = new Array(values.length);\n    const reads = new Array(values.length);");
testWorker=replaceOne(testWorker,
    "const value = values[i];",
    "const value = values[i];\n        const stream = { value: value ^ 0xa5a5a5a5a5a5a5a5n };");
testWorker=replaceOne(testWorker,
    "            if (offset === buffer.length) {\n                crypto.getRandomValues(buffer);\n                offset = 0;\n            }\n            const word = buffer[offset++];",
    "            let word;\n            if (deterministic) word = splitmix64(stream);\n            else {\n                if (offset === buffer.length) {\n                    crypto.getRandomValues(buffer);\n                    offset = 0;\n                }\n                word = buffer[offset++];\n            }");
testWorker=replaceOne(testWorker,
    "blocks[i] = formatOutput(step.value);",
    "blocks[i] = formatOutput(step.value);\n        reads[i] = recordSupplements;");
testWorker=replaceOne(testWorker,
    "return { start, blocks, supplements };",
    "return { start, blocks, supplements, reads };");
testWorker+=String.raw
const server=createServer((req,res)=>{
    const pathname=new URL(req.url,"http://127.0.0.1").pathname;
    const special=pathname==="/srg_parallel.js"?testParallel:pathname==="/srg_worker.js"?testWorker:null;
    if(special!==null) {
        res.writeHead(200,{"Content-Type":"text/javascript","Content-Length":Buffer.byteLength(special),"Cache-Control":"no-store"});
        res.end(special);
        return;
    }
    const path=resolve(root,pathname.slice(1)||"revgeo.html");
    if(!path.startsWith(root+sep)){res.writeHead(403);res.end();return;}
    try{
        const stat=statSync(path);
        if(!stat.isFile())throw Error("not file");
        const type=path.endsWith(".js")?"text/javascript":path.endsWith(".html")?"text/html":"application/octet-stream";
        res.writeHead(200,{"Content-Type":type,"Content-Length":stat.size,"Cache-Control":"no-store"});
        createReadStream(path).pipe(res);
    }catch{res.writeHead(404);res.end();}
});
await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
const origin="http://127.0.0.1:"+server.address().port;
const results=[];
try{
    for(const [name,engine] of [["chromium",chromium],["firefox",firefox],["webkit",webkit]]){
        const browser=await engine.launch({headless:true});
        try{
            for(const workerCount of [0,1,2,4]){
                const context=await browser.newContext();
                const network={requests:0,bytes:0};
                context.on("response",response=>{
                    if(response.url().endsWith(".bin")){
                        network.requests++;
                        network.bytes+=Number(response.headers()["content-length"]||0);
                    }
                });
                try{
                    const page=await context.newPage();
                    page.setDefaultTimeout(300000);
                    await page.goto(origin+"/revgeo.html",{waitUntil:"domcontentloaded"});
                    const measured=await page.evaluate(async ({workerCount})=>{
                        const {convertNumber,formatOutput}=await import("./srg_random.js?v=1");
                        const {sampleKorea}=await import("./korea_boundary.js?v=6");
                        const {convertParallel}=await import("./srg_parallel.js?v=1");
                        const mask=(1n<<64n)-1n;
                        function next(state){
                            state.value=(state.value+0x9e3779b97f4a7c15n)&mask;
                            let word=state.value;
                            word=((word^(word>>30n))*0xbf58476d1ce4e5b9n)&mask;
                            word=((word^(word>>27n))*0x94d049bb133111ebn)&mask;
                            return(word^(word>>31n))&mask;
                        }
                        const inputs=new Array(10000);
                        const stream={value:0x123456789abcdef0n};
                        for(let i=0;i<inputs.length;i++)inputs[i]=next(stream);
                        async function single(){
                            const blocks=[];
                            let supplements=0;
                            const reads=[];
                            for(const number of inputs){
                                const state={value:number^0xa5a5a5a5a5a5a5a5n};
                                const it=convertNumber(number,sampleKorea);
                                let step=await it.next(),i=0;
                                while(!step.done&&i<1024){
                                    step=await it.next({value:next(state),bitsRemaining:64});
                                    i++;
                                }
                                if(!step.done)throw Error("serial retries exceeded");
                                blocks.push(formatOutput(step.value));
                                reads.push(i);
                                supplements+=i;
                            }
                            return {blocks,supplements,reads};
                        }
                        const bench=async fn=>{
                            const t=performance.now();
                            const result=await fn();
                            return {durationMs:performance.now()-t,...result};
                        };
                        const samples=[];
                        let reference=null;
                        for(let repeat=0;repeat<5;repeat++){
                            const result=await bench(()=>workerCount?convertParallel(inputs,{workers:workerCount,deterministic:true}):single());
                            if(reference){
                                for(let i=0;i<inputs.length;i++)if(reference.blocks[i]!==result.blocks[i])throw Error("warm run output changed at "+i);
                            }else reference=result;
                            samples.push(result.durationMs);
                        }
                        const view=new DataView(new ArrayBuffer(8));
                        const bits=x=>{view.setFloat64(0,x,false);return view.getBigUint64(0,false).toString(16).padStart(16,"0");};
                        const resultLines=reference.blocks.map((block,i)=>{
                            const korea=/대한민국 위경도: ([^,]+), ([^\n]+)/.exec(block);
                            const world=/세계 위경도: ([^,]+), ([^\n]+)/.exec(block);
                            if(!korea||!world)throw Error("missing coordinate fields");
                            const supp={value:inputs[i]^0xa5a5a5a5a5a5a5a5n};
                            return [bits(Number(korea[1])),bits(Number(korea[2])),bits(Number(world[1])),bits(Number(world[2])),String(reference.reads[i])].join(" ");
                        });
                        const sort=samples.slice().sort((a,b)=>a-b);
                        const expectedSupplemental=reference.supplements;
                        return {workerCount,medianMs:sort[2],minMs:sort[0],maxMs:sort[4],samplesMs:samples,substituteWords:expectedSupplemental,results:resultLines,first:reference.blocks[0],last:reference.blocks.at(-1)};
                    },{workerCount});
                    let mismatches=0;
                    const examples=[];
                    for(let i=0;i<10000;i++){
                        const e=expected[i].split(" ");
                        if(measured.results[i]!==e.join(" ")){
                            mismatches++;
                            if(examples.length<4)examples.push({at:i,expected:e.join(" "),actual:measured.results[i]});
                        }
                    }
                    console.log("WORKER_BENCH "+JSON.stringify({browser:name,version:browser.version(),workerCount,medianMs:measured.medianMs,minMs:measured.minMs,maxMs:measured.maxMs,samplesMs:measured.samplesMs,substituteWords:measured.substituteWords,mismatches,examples,network}));
                    if(mismatches)throw Error("Rust/browser parity failure with "+workerCount+" workers in "+name);
                    results.push({name,workerCount,medianMs:measured.medianMs,mismatches});
                }finally{await context.close();}
            }
        }finally{await browser.close();}
    }
}finally{await new Promise(resolve=>server.close(resolve));}
console.log("WORKER_FINAL "+JSON.stringify(results));
