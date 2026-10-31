import {createReadStream,statSync} from "node:fs";
import {createServer} from "node:http";
import {resolve,sep} from "node:path";
import {chromium,firefox,webkit} from "playwright";
const root=process.cwd();
const server=createServer((req,res)=>{
    const path=resolve(root,new URL(req.url,"http://127.0.0.1").pathname.slice(1)||"revgeo.html");
    if(!path.startsWith(root+sep)){res.writeHead(403);res.end();return;}
    try{
        const stat=statSync(path);
        if(!stat.isFile())throw Error("not file");
        res.writeHead(200,{"Content-Length":stat.size,"Cache-Control":"no-store","Content-Type":path.endsWith(".html")?"text/html":path.endsWith(".js")?"text/javascript":"application/octet-stream"});
        createReadStream(path).pipe(res);
    }catch{res.writeHead(404);res.end();}
});
await new Promise(done=>server.listen(0,"127.0.0.1",done));
const url="http://127.0.0.1:"+server.address().port+"/revgeo.html";
const tokens=Array.from({length:10000},(_,i)=>String(BigInt(i)*10101010101n+987654321n));
async function run(page,count){
    await page.locator("#manual-tab").click();
    await page.evaluate(words=>{
        const input=document.getElementById("num64");
        input.value=words.join("\n");
        input.dispatchEvent(new Event("input",{bubbles:true}));
        document.getElementById("auto-supp").checked=true;
        document.getElementById("runBtn").click();
    },tokens.slice(0,count));
}
async function done(page,count){
    await page.waitForFunction(count=>{
        const error=document.querySelector("#error")?.textContent;
        if(error)throw Error(error);
        return (document.querySelector("#meta")?.textContent||"").includes("변환 완료 · "+count.toLocaleString("ko-KR")+"개");
    },count,{timeout:120000});
}
try{
    for(const [name,engine] of [["chromium",chromium],["firefox",firefox],["webkit",webkit]]){
        const browser=await engine.launch({headless:true});
        try{
            for(const blockMode of ["missing","denied"]){
                const context=await browser.newContext();
                if(blockMode==="missing")
                    await context.addInitScript(()=>{Object.defineProperty(globalThis,"Worker",{value:undefined,configurable:true});});
                else
                    await context.addInitScript(()=>{Object.defineProperty(globalThis,"Worker",{value:class WorkerDenied{constructor(){throw new DOMException("denied","SecurityError");}},configurable:true});});
                try{
                    const page=await context.newPage();
                    const launched=[];
                    page.on("worker",worker=>launched.push(worker));
                    const errors=[];
                    page.on("pageerror",error=>errors.push(error.message));
                    await page.goto(url,{waitUntil:"domcontentloaded"});
                    await run(page,700);
                    await done(page,700);
                    const meta=await page.locator("#meta").textContent();
                    console.log("FALLBACK "+JSON.stringify({browser:name,mode:blockMode,workers:launched.length,meta,pageErrors:errors}));
                    if(launched.length||errors.length)throw Error("Single-thread fallback failed");
                }finally{await context.close();}
            }
            const context=await browser.newContext();
            try{
                const page=await context.newPage();
                const launched=[];
                let closed=0;
                const errors=[];
                page.on("pageerror",error=>errors.push(error.message));
                page.on("worker",worker=>{
                    launched.push(worker);
                    worker.on("close",()=>closed++);
                });
                await page.goto(url,{waitUntil:"domcontentloaded"});
                const firstWorker=page.waitForEvent("worker",{timeout:30000});
                await run(page,10000);
                await firstWorker;
                await page.locator("#clearBtn").click();
                await page.waitForFunction(()=>document.getElementById("meta")?.textContent===""&&!document.getElementById("runBtn").disabled,{timeout:30000});
                await page.waitForFunction(()=>document.querySelector("#output .code-content")?.textContent==="",null,{timeout:30000});
                await page.waitForFunction(()=>document.getElementById("runBtn").disabled===false,{timeout:30000});
                await page.evaluate(()=>{
                    const input=document.getElementById("num64");
                    input.value="123456789";
                    input.dispatchEvent(new Event("input",{bubbles:true}));
                    document.getElementById("runBtn").click();
                });
                await done(page,1);
                await page.waitForTimeout(250);
                const meta=await page.locator("#meta").textContent();
                console.log("CANCEL "+JSON.stringify({browser:name,workersLaunched:launched.length,workersClosed:closed,meta,pageErrors:errors}));
                if(!launched.length||closed<launched.length||errors.length)throw Error("Worker cleanup or rerun after cancellation failed");
            }finally{await context.close();}
        }finally{await browser.close();}
    }
}finally{await new Promise(done=>server.close(done));}
