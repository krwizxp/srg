import {execFileSync} from "node:child_process";
import {readFileSync,createReadStream,statSync} from "node:fs";
import {createServer} from "node:http";
import {resolve,sep} from "node:path";
import {chromium,firefox,webkit} from "playwright";

const baseline=execFileSync("git",["show","57ddaa4f63584797ae8d22a64fb6c01149ae59e6:revgeo.html"],{encoding:"utf8"});
const last=baseline.lastIndexOf("</script>");
if(last<0)throw Error("Missing original inline conversion script");
const baselinePage=baseline.slice(0,last)+"\nwindow.__srgOriginal = { convertNumber, formatOutput };\n"+baseline.slice(last);
const root=process.cwd();
const server=createServer((req,res)=>{
    const pathname=new URL(req.url,"http://127.0.0.1").pathname;
    if(pathname==="/baseline/revgeo.html"){
        res.writeHead(200,{"Content-Type":"text/html; charset=utf-8","Cache-Control":"no-store"});
        res.end(baselinePage);
        return;
    }
    const path=resolve(root,(pathname.startsWith("/baseline/")?pathname.slice(10):pathname.slice(1))||"revgeo.html");
    if(!path.startsWith(root+sep)){res.writeHead(403);res.end();return;}
    try{
        const stat=statSync(path);
        if(!stat.isFile())throw Error("not file");
        res.writeHead(200,{"Content-Length":stat.size,"Cache-Control":"no-store","Content-Type":path.endsWith(".html")?"text/html":path.endsWith(".js")?"text/javascript":"application/octet-stream"});
        createReadStream(path).pipe(res);
    }catch{res.writeHead(404);res.end();}
});
await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
const rootUrl="http://127.0.0.1:"+server.address().port;
try{
    for(const [name,engine] of [["chromium",chromium],["firefox",firefox],["webkit",webkit]]){
        const browser=await engine.launch({headless:true});
        try{
            const versions=[];
            for(const version of ["baseline","candidate"]){
                const page=await browser.newPage();
                const errors=[];
                page.on("pageerror",e=>errors.push(e.message));
                try{
                    await page.goto(rootUrl+(version==="baseline"?"/baseline/revgeo.html":"/revgeo.html"),{waitUntil:"domcontentloaded"});
                    if(version==="baseline")
                        await page.waitForFunction(()=>typeof window.__srgOriginal?.convertNumber==="function");
                    const result=await page.evaluate(async ({version})=>{
                        const old=window.__srgOriginal;
                        const current=version==="baseline"?old:await import("./srg_random.js?v=1");
                        const {sampleKorea}=await import("./korea_boundary.js?v=6");
                        const mask=(1n<<64n)-1n;
                        function next(state){
                            state.value=(state.value+0x9e3779b97f4a7c15n)&mask;
                            let word=state.value;
                            word=((word^(word>>30n))*0xbf58476d1ce4e5b9n)&mask;
                            word=((word^(word>>27n))*0x94d049bb133111ebn)&mask;
                            return(word^(word>>31n))&mask;
                        }
                        const seed={value:0x123456789abcdef0n};
                        const blocks=[];
                        let reads=0;
                        const started=performance.now();
                        for(let i=0;i<10000;i++){
                            const word=next(seed);
                            const supplementary={value:word^0xa5a5a5a5a5a5a5a5n};
                            const iter=current.convertNumber(word,sampleKorea);
                            let response=await iter.next(),used=0;
                            while(!response.done&&used<1024){
                                used++;
                                response=await iter.next({value:next(supplementary),bitsRemaining:64});
                            }
                            if(!response.done)throw Error("Unexpected supplemental exhaustion");
                            blocks.push(current.formatOutput(response.value));
                            reads+=used;
                        }
                        const payload=blocks.join("\n")+"\n";
                        const digest=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(payload)));
                        return {elapsedMs:performance.now()-started,sha256:Array.from(digest).map(x=>x.toString(16).padStart(2,"0")).join(""),bytes:new TextEncoder().encode(payload).length,reads,first:blocks[0],last:blocks.at(-1),count:blocks.length};
                    },{version});
                    versions.push({version,...result,pageErrors:errors});
                }finally{await page.close();}
            }
            const [before,after]=versions;
            const same=before.sha256===after.sha256 && before.bytes===after.bytes && before.reads===after.reads&&before.count===after.count&&before.first===after.first&&before.last===after.last;
            console.log("FULL_OUTPUT_PARITY "+JSON.stringify({browser:name,version:browser.version(),count:before.count,equal:same,baselineSHA256:before.sha256,candidateSHA256:after.sha256,bytes:before.bytes,supplementalReads:before.reads,baselineMs:before.elapsedMs,candidateMs:after.elapsedMs,pageErrors:versions.map(v=>v.pageErrors)}));
            if(!same||versions.some(v=>v.pageErrors.length))throw Error("Extracted shared module changed full output for "+name);
        }finally{await browser.close();}
    }
}finally{await new Promise(resolve=>server.close(resolve));}
