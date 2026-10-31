import {readFileSync,createReadStream,statSync} from "node:fs";
import {createServer} from "node:http";
import {resolve,sep} from "node:path";
import {chromium,firefox,webkit} from "playwright";

const COUNT=10000;
const REPS=5;
const SALT=0xa5a5a5a5a5a5a5a5n;
const MAX=(1n<<64n)-1n;
const expectedCoords=readFileSync("validation/bench10000_rust_coords.txt","utf8").trim().split(/\r?\n/);
const expectedFull=readFileSync("validation/bench10000_rust_full.txt","utf8").trim().split(/\r?\n/);
if(expectedCoords.length!==COUNT||expectedFull.length!==COUNT)throw Error("Rust provided incorrect sample count");
const root=process.cwd();
const html=readFileSync("revgeo.html","utf8");
const scriptTag=html.lastIndexOf("</script>");
if(scriptTag<0||!html.includes("async function* convertNumber("))throw Error("SRG web module missing");
const htmlWithProbe=html.slice(0,scriptTag)+"globalThis.__srgBench = {convertNumber,formatOutput};\n"+html.slice(scriptTag);
const server=createServer((req,res)=>{
    const relative=new URL(req.url,"http://127.0.0.1").pathname.slice(1)||"revgeo.html";
    if(relative==="revgeo.html"){
        res.writeHead(200,{"Content-Type":"text/html;charset=utf-8","Cache-Control":"no-store"});
        res.end(htmlWithProbe);
        return;
    }
    const file=resolve(root,relative);
    if(!file.startsWith(root+sep)){res.writeHead(403);res.end();return;}
    try{
        const stat=statSync(file);
        if(!stat.isFile())throw Error("not file");
        const mime=file.endsWith(".js")?"text/javascript":"application/octet-stream";
        res.writeHead(200,{"Content-Type":mime,"Content-Length":stat.size,"Cache-Control":"no-store"});
        createReadStream(file).pipe(res);
    }catch{res.writeHead(404);res.end();}
});
await new Promise(ok=>server.listen(0,"127.0.0.1",ok));
const url="http://127.0.0.1:"+server.address().port+"/revgeo.html";
function compare(actual,expected,label,engine){
    let mismatch=0;
    const examples=[];
    for(let i=0;i<expected.length;i++){
        if(actual[i]!==expected[i]){
            mismatch++;
            if(examples.length<3)examples.push({at:i,expected:expected[i],actual:actual[i]});
        }
    }
    console.log("PARITY "+JSON.stringify({engine,kind:label,count:expected.length,mismatches:mismatch,examples}));
    return mismatch;
}
let invalid=0;
try{
    for(const [engineName,engine] of [["chromium",chromium],["firefox",firefox],["webkit",webkit]]){
        const browser=await engine.launch({headless:true});
        try{
            const context=await browser.newContext();
            const page=await context.newPage();
            page.setDefaultTimeout(600000);
            const request={parts:0,bytes:0};
            page.on("response",response=>{
                if(response.url().endsWith(".bin")){
                    request.parts++;
                    request.bytes+=Number(response.headers()["content-length"]||0);
                }
            });
            await page.goto(url,{waitUntil:"domcontentloaded"});
            await page.waitForFunction(()=>typeof globalThis.__srgBench?.convertNumber==="function");
            const results=await page.evaluate(async ({COUNT,REPS})=>{
                const mask=(1n<<64n)-1n;
                const salt=0xa5a5a5a5a5a5a5a5n;
                function splitmix(state){
                    state.value=(state.value+0x9e3779b97f4a7c15n)&mask;
                    let word=state.value;
                    word=((word^(word>>30n))*0xbf58476d1ce4e5b9n)&mask;
                    word=((word^(word>>27n))*0x94d049bb133111ebn)&mask;
                    return(word^(word>>31n))&mask;
                }
                const state={value:0x123456789abcdef0n};
                const words=Array.from({length:COUNT},()=>splitmix(state));
                const view=new DataView(new ArrayBuffer(8));
                function bits(value){
                    view.setFloat64(0,value,false);
                    return view.getBigUint64(0,false);
                }
                function hex(value){return value.toString(16).padStart(16,"0");}
                const tImport=performance.now();
                const sampler=await import("./korea_boundary.js?benchmark10k=1");
                const importMs=performance.now()-tImport;
                async function doCoords(capture){
                    let checksum=0n;
                    let supplements=0;
                    const rows=capture?[]:null;
                    for(const word of words){
                        const supp={value:word^salt};
                        const iter=sampler.sampleKorea(word);
                        let step=await iter.next();
                        let reads=0;
                        while(!step.done&&reads<1024){
                            reads++;
                            step=await iter.next({value:splitmix(supp),bitsRemaining:64});
                        }
                        if(!step.done)throw Error("coordinate attempts exceeded");
                        const [lat,lon]=step.value;
                        const latBits=bits(lat),lonBits=bits(lon);
                        checksum=(checksum+((latBits^lonBits^BigInt(reads))&mask))&mask;
                        supplements+=reads;
                        if(capture)rows.push(hex(latBits)+" "+hex(lonBits)+" "+reads);
                    }
                    return {checksum:hex(checksum),supplements,rows};
                }
                async function doFull(capture){
                    let checksum=0n;
                    let supplements=0;
                    const rows=capture?[]:null;
                    for(const word of words){
                        const supp={value:word^salt};
                        const conversion=globalThis.__srgBench.convertNumber(word,sampler.sampleKorea);
                        let step=await conversion.next();
                        let reads=0;
                        while(!step.done&&reads<1024){
                            reads++;
                            step=await conversion.next({value:splitmix(supp),bitsRemaining:64});
                        }
                        if(!step.done)throw Error("full conversion attempts exceeded");
                        const result=step.value;
                        const lat=bits(result.korea[0]),lon=bits(result.korea[1]);
                        const wlat=bits(result.world[0]),wlon=bits(result.world[1]);
                        checksum=(checksum+((lat^lon^wlat^wlon^BigInt(reads))&mask))&mask;
                        supplements+=reads;
                        if(capture)rows.push(hex(lat)+" "+hex(lon)+" "+hex(wlat)+" "+hex(wlon)+" "+reads);
                    }
                    return {checksum:hex(checksum),supplements,rows};
                }
                function median(a){return a.slice().sort((x,y)=>x-y)[Math.floor(a.length/2)];}
                async function measure(fn){
                    const t0=performance.now();
                    const cold=await fn(false);
                    const coldMs=performance.now()-t0;
                    const reference=await fn(true);
                    if(cold.checksum!==reference.checksum||cold.supplements!==reference.supplements)throw Error("cache changed results");
                    const warmup=await fn(false);
                    if(warmup.checksum!==cold.checksum)throw Error("warmup changed results");
                    const repetitions=[];
                    for(let repeat=0;repeat<REPS;repeat++){
                        const start=performance.now();
                        const result=await fn(false);
                        const elapsed=performance.now()-start;
                        if(result.checksum!==cold.checksum||result.supplements!==cold.supplements)throw Error("replay changed results");
                        repetitions.push(elapsed);
                    }
                    return {coldMs,medianMs:median(repetitions),minMs:Math.min(...repetitions),maxMs:Math.max(...repetitions),timesMs:repetitions,supplementalReads:cold.supplements,checksum:cold.checksum,rows:reference.rows};
                }
                const coord=await measure(doCoords);
                const full=await measure(doFull);
                return {importMs,coord,full};
            },{COUNT,REPS});
            invalid+=compare(results.coord.rows,expectedCoords,"coords",engineName);
            invalid+=compare(results.full.rows,expectedFull,"full",engineName);
            delete results.coord.rows;
            delete results.full.rows;
            console.log("BROWSER_BENCH "+JSON.stringify({engine:engineName,version:browser.version(),n:COUNT,...results,network:request}));
            const ui=await context.newPage();
            ui.setDefaultTimeout(600000);
            await ui.goto(url,{waitUntil:"domcontentloaded"});
            await ui.waitForFunction(()=>typeof globalThis.__srgBench?.convertNumber==="function");
            await ui.locator("#manual-tab").click();
            const run=await ui.evaluate(({COUNT})=>{
                const mask=(1n<<64n)-1n;
                let state=0x123456789abcdef0n;
                function next(){
                    state=(state+0x9e3779b97f4a7c15n)&mask;
                    let word=state;
                    word=((word^(word>>30n))*0xbf58476d1ce4e5b9n)&mask;
                    word=((word^(word>>27n))*0x94d049bb133111ebn)&mask;
                    return(word^(word>>31n))&mask;
                }
                const words=Array.from({length:COUNT},()=>next().toString());
                const area=document.getElementById("num64");
                area.value=words.join("\n");
                area.dispatchEvent(new Event("input",{bubbles:true}));
                document.getElementById("auto-supp").checked=true;
                globalThis.__uiStart=performance.now();
                document.getElementById("runBtn").click();
                return {bytes:area.value.length};
            },{COUNT});
            await ui.waitForFunction(()=>{
                const info=document.getElementById("meta")?.textContent||"";
                const error=document.getElementById("error")?.textContent||"";
                if(error)throw Error(error);
                return info.includes("변환 완료")&&info.includes("10,000");
            },{timeout:600000});
            const done=await ui.evaluate(()=>({
                elapsedMs:performance.now()-globalThis.__uiStart,
                status:document.getElementById("meta").textContent,
                previewLength:document.querySelector("#output .code-content").textContent.length
            }));
            console.log("UI_BENCH "+JSON.stringify({engine:engineName,version:browser.version(),n:COUNT,inputChars:run.bytes,...done}));
            await context.close();
        }finally{await browser.close();}
    }
}finally{await new Promise(ok=>server.close(ok));}
if(invalid)throw Error("Rust/web outputs diverged on "+invalid+" records");
