import {readFileSync,createReadStream,statSync} from "node:fs";
import {createServer} from "node:http";
import {resolve,sep} from "node:path";
import {chromium,firefox,webkit} from "playwright";
const root=process.cwd();
const server=createServer((request,response)=>{
    const target=resolve(root,new URL(request.url,"http://127.0.0.1").pathname.slice(1)||"revgeo.html");
    if(!target.startsWith(root+sep)){response.writeHead(403);response.end();return;}
    try{
        const st=statSync(target);
        if(!st.isFile())throw Error("not file");
        response.writeHead(200,{"Content-Type":target.endsWith(".html")?"text/html;charset=utf-8":target.endsWith(".js")?"text/javascript":"application/octet-stream","Content-Length":st.size,"Cache-Control":"no-store"});
        createReadStream(target).pipe(response);
    }catch{response.writeHead(404);response.end();}
});
await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
const url="http://127.0.0.1:"+server.address().port+"/revgeo.html";
const n=10000;
function next64(state){
    const mask=(1n<<64n)-1n;
    state.value=(state.value+0x9e3779b97f4a7c15n)&mask;
    let x=state.value;
    x=((x^(x>>30n))*0xbf58476d1ce4e5b9n)&mask;
    x=((x^(x>>27n))*0x94d049bb133111ebn)&mask;
    return (x^(x>>31n))&mask;
}
const inputs=Array.from({length:n},()=>null);
let seed={value:0x123456789abcdef0n};
for(let i=0;i<n;i++)inputs[i]=next64(seed);
let errors=0;
try{
    for(const [name,engine] of [["chromium",chromium],["firefox",firefox],["webkit",webkit]]){
        const browser=await engine.launch({headless:true});
        try{
            const context=await browser.newContext({acceptDownloads:true});
            const page=await context.newPage();
            page.setDefaultTimeout(120000);
            const pageErrors=[];
            const workers=[];
            page.on("pageerror",error=>pageErrors.push(error.message));
            page.on("worker",worker=>workers.push(worker));
            await page.goto(url,{waitUntil:"domcontentloaded"});
            await page.locator("#manual-tab").click();
            const run=await page.evaluate(inputs=>{
                const area=document.getElementById("num64");
                area.value=inputs.map(value=>String(value)).join("\n");
                area.dispatchEvent(new Event("input",{bubbles:true}));
                document.getElementById("auto-supp").checked=true;
                const started=performance.now();
                document.getElementById("runBtn").click();
                return {inputChars:area.value.length,started};
            },inputs.map(x=>x.toString()));
            await page.waitForFunction(()=>{
                const error=document.querySelector("#error")?.textContent;
                if(error)throw Error(error);
                return (document.querySelector("#meta")?.textContent||"").includes("변환 완료");
            });
            const result=await page.evaluate(started=>({
                elapsedMs:performance.now()-started,
                meta:document.querySelector("#meta")?.textContent,
                preview:document.querySelector("#output .code-content")?.textContent,
                saveEnabled:!document.getElementById("saveOutput").disabled,
            }),run.started);
            const [download]=await Promise.all([page.waitForEvent("download"),page.locator("#saveOutput").click()]);
            const saved=readFileSync(await download.path(),"utf8").replace(/^\uFEFF/,"");
            const matches=[...saved.matchAll(/^64비트 난수: ([0-9]+)/gm)];
            let orderMismatch=0;
            for(let i=0;i<inputs.length&&i<matches.length;i++)
                if(matches[i][1]!==inputs[i].toString())orderMismatch++;
            const reported={browser:name,version:browser.version(),n,elapsedMs:result.elapsedMs,meta:result.meta,workersSpawned:workers.length,savedBytes:Buffer.byteLength(saved),records:matches.length,orderMismatch,previewRecords:(result.preview.match(/64비트 난수:/g)||[]).length,saveEnabled:result.saveEnabled,pageErrors};
            console.log("UI_WORKERS "+JSON.stringify(reported));
            if(workers.length<2||workers.length>4||matches.length!==n||orderMismatch!==0||pageErrors.length||!result.saveEnabled||!result.meta.includes("10,000")){errors++;throw Error(name+" UI production parallel integration failed");}
            const manual=await context.newPage();
            const manualWorkers=[];
            manual.on("worker",worker=>manualWorkers.push(worker));
            manual.on("pageerror",error=>{errors++;console.log("PAGE_ERROR "+name+" "+error.message);});
            await manual.goto(url,{waitUntil:"domcontentloaded"});
            await manual.locator("#manual-tab").click();
            const manualInputs=inputs.slice(0,100);
            const suppSeed={value:0x44556677aabbccddn};
            const supplements=Array.from({length:6000},()=>next64(suppSeed));
            const expected=await manual.evaluate(async ({inputs,supplements})=>{
                const {convertNumber,formatOutput}=await import("./srg_random.js?v=1");
                const {sampleKorea}=await import("./korea_boundary.js?v=6");
                let at=0;
                const rows=[];
                for(const value of inputs){
                    const iter=convertNumber(BigInt(value),sampleKorea);
                    let s=await iter.next();
                    while(!s.done){
                        if(at>=supplements.length)throw Error("reference supplements exhausted");
                        s=await iter.next({value:BigInt(supplements[at++]),bitsRemaining:64});
                    }
                    rows.push(formatOutput(s.value));
                }
                return {text:rows.join("\n")+"\n",used:at};
            },{inputs:manualInputs.map(String),supplements:supplements.map(String)});
            await manual.evaluate(({inputs,supplements})=>{
                const num=document.getElementById("num64");
                const supp=document.getElementById("suppValues");
                num.value=inputs.join("\n");
                supp.value=supplements.join("\n");
                num.dispatchEvent(new Event("input",{bubbles:true}));
                document.getElementById("auto-supp").checked=true;
                document.getElementById("runBtn").click();
            },{inputs:manualInputs.map(String),supplements:supplements.map(String)});
            await manual.waitForFunction(()=>{
                const error=document.querySelector("#error")?.textContent;
                if(error)throw Error(error);
                return (document.querySelector("#meta")?.textContent||"").includes("변환 완료");
            });
            const [manualDownload]=await Promise.all([manual.waitForEvent("download"),manual.locator("#saveOutput").click()]);
            const savedManual=readFileSync(await manualDownload.path(),"utf8").replace(/^\uFEFF/,"");
            const manualOk=savedManual===expected.text;
            console.log("UI_MANUAL "+JSON.stringify({browser:name,n:100,workersSpawned:manualWorkers.length,manualOk,expectedSupplements:expected.used}));
            if(!manualOk||manualWorkers.length){errors++;throw Error(name+" manual supplied supplement ordering changed");}
            await context.close();
        }finally{await browser.close();}
    }
}finally{await new Promise(resolve=>server.close(resolve));}
if(errors)throw Error("worker UI integration failures: "+errors);
