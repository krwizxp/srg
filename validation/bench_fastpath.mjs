import {readFileSync,createReadStream,statSync} from "node:fs";
import {createServer} from "node:http";
import {resolve,sep} from "node:path";
import {chromium,firefox,webkit} from "playwright";

const source=readFileSync("korea_boundary.js","utf8");
const earlyOriginal="++boundary.mixedChecks < 12";
const fastOriginal="            if (await covers(boundary, point)) return point;";
if(!source.includes(earlyOriginal)||!source.includes(fastOriginal))throw Error("sampler hook changed");
const early=source.replace(earlyOriginal,"++boundary.mixedChecks < 1");
const fastCode="            const tile = tilePosition(boundary, point);\n            const state = boundary.data.getUint32(128 + tile * 12, true);\n            if (state === 1 || (state === 2 && await covers(boundary, point))) return point;";
const fast=source.replace(fastOriginal,fastCode);
const both=early.replace(fastOriginal,fastCode);
const modes=new Map([["baseline",source],["early",early],["fast",fast],["both",both]]);
const root=process.cwd();
const server=createServer((req,res)=>{
    const path=new URL(req.url,"http://127.0.0.1").pathname;
    const pieces=path.split("/");
    const mode=pieces[1],assetPath="/"+pieces.slice(2).join("/");
    if(assetPath==="/korea_boundary.js"&&modes.has(mode)){
        const body=modes.get(mode);
        res.writeHead(200,{"Content-Type":"text/javascript","Content-Length":Buffer.byteLength(body),"Cache-Control":"no-store"});res.end(body);return;
    }
    const safePath=assetPath==="/"?"revgeo.html":assetPath.slice(1);
    const file=resolve(root,safePath);
    if(!file.startsWith(root+sep)){res.writeHead(403);res.end();return;}
    try{
        const st=statSync(file);
        if(!st.isFile())throw Error("not file");
        res.writeHead(200,{"Content-Length":st.size,"Cache-Control":"no-store","Content-Type":file.endsWith(".html")?"text/html":"application/octet-stream"});
        createReadStream(file).pipe(res);
    }catch{res.writeHead(404);res.end();}
});
await new Promise(done=>server.listen(0,"127.0.0.1",done));
const origin="http://127.0.0.1:"+server.address().port;
try{
    for(const [browserName,engine] of [["chromium",chromium],["firefox",firefox],["webkit",webkit]]){
        const browser=await engine.launch({headless:true});
        try{
            const results={};
            const order=browserName==="chromium"?["baseline","early","fast","both"]:
                browserName==="firefox"?["both","fast","early","baseline"]:["fast","baseline","both","early"];
            for(const mode of order){
                const context=await browser.newContext();
                const count={binRequests:0,binBytes:0};
                context.on("response",response=>{
                    if(response.url().includes(".bin")){
                        count.binRequests++;
                        count.binBytes+=Number(response.headers()["content-length"]||0);
                    }
                });
                try{
                    const page=await context.newPage();
                    await page.goto(origin+"/"+mode+"/revgeo.html",{waitUntil:"domcontentloaded"});
                    const data=await page.evaluate(async mode=>{
                        const mod=await import("./korea_boundary.js?bench="+mode);
                        const mask=(1n<<64n)-1n;
                        function splitmix(s){
                            s.value=(s.value+0x9e3779b97f4a7c15n)&mask;
                            let x=s.value;
                            x=((x^(x>>30n))*0xbf58476d1ce4e5b9n)&mask;
                            x=((x^(x>>27n))*0x94d049bb133111ebn)&mask;
                            return(x^(x>>31n))&mask;
                        }
                        const state={value:0x123456789abcdef0n},coords=[];
                        let supplementalReads=0;
                        const t=performance.now();
                        for(let i=0;i<650;i++){
                            const word=splitmix(state);
                            const supplement={value:word^0xa5a5a5a5a5a5a5a5n};
                            const iter=mod.sampleKorea(word);
                            let outcome=await iter.next(),reads=0;
                            while(!outcome.done&&reads<1024){
                                reads++;outcome=await iter.next({value:splitmix(supplement),bitsRemaining:64});
                            }
                            if(!outcome.done)throw Error("too many tries");
                            const [lat,lon]=outcome.value;
                            supplementalReads+=reads;
                            coords.push(lat.toString()+","+lon.toString()+":"+reads);
                        }
                        return {ms:performance.now()-t,supplementalReads,coords};
                    },mode);
                    results[mode]={durationMs:data.ms,supplementalReads:data.supplementalReads,coords:data.coords,...count};
                }finally{await context.close();}
            }
            for(const variant of ["early","fast","both"]){
                const baseline=results.baseline.coords,now=results[variant].coords;
                let different=0;for(let i=0;i<baseline.length;i++)if(baseline[i]!==now[i])different++;
                results[variant].mismatches=different;
                if(different)throw Error(variant+" changed sampler");
            }
            console.log("FASTPATH "+JSON.stringify({browser:browserName,version:browser.version(),results:Object.fromEntries(Object.entries(results).map(([key,x])=>[key,{durationMs:x.durationMs,supplementalReads:x.supplementalReads,binRequests:x.binRequests,binBytes:x.binBytes,mismatches:x.mismatches||0}]))}));
        }finally{await browser.close();}
    }
}finally{await new Promise(done=>server.close(done));}
