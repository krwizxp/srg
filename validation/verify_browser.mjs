import {readFileSync,createReadStream,statSync} from "node:fs";
import {createServer} from "node:http";
import {resolve,sep} from "node:path";
import {execFileSync} from "node:child_process";
import {chromium,firefox,webkit} from "playwright";
const root=process.cwd();
const baseline=execFileSync("git",["show","d749b386c7a0aad665cb8686ddbd78abcc894cb8:korea_boundary.js"]);
const server=createServer((req,res)=>{
    const path=new URL(req.url,"http://127.0.0.1").pathname;
    if(path==="/baseline/korea_boundary.js"){
        res.writeHead(200,{"Content-Type":"text/javascript","Content-Length":baseline.length,"Cache-Control":"no-store"});
        res.end(baseline);
        return;
    }
    let rel=path.startsWith("/baseline/data/")?path.slice("/baseline".length):path;
    if(rel==="/") rel="/revgeo.html";
    const file=resolve(root,"."+rel);
    if(!file.startsWith(root+sep)){
        res.writeHead(403);res.end();return;
    }
    try{
        const stat=statSync(file);
        if(!stat.isFile())throw Error("not a file");
        const mime=rel.endsWith(".js")?"text/javascript":rel.endsWith(".html")?"text/html":"application/octet-stream";
        res.writeHead(200,{"Content-Type":mime,"Content-Length":stat.size,"Cache-Control":"no-store"});
        createReadStream(file).pipe(res);
    }catch{
        res.writeHead(404);res.end();
    }
});
await new Promise((yes,no)=>server.listen(0,"127.0.0.1",e=>e?no(e):yes()));
const url="http://127.0.0.1:"+server.address().port+"/revgeo.html";
const results=[];
try{
    for(const [engineName,engine] of [["chromium",chromium],["firefox",firefox],["webkit",webkit]]){
        const browser=await engine.launch({headless:true});
        try{
            const page=await browser.newPage();
            await page.goto(url,{waitUntil:"domcontentloaded"});
            const result=await page.evaluate(async()=>{
                const oldModule=await import("/baseline/korea_boundary.js?baseline=1");
                const newModule=await import("/korea_boundary.js?optimized=1");
                const mask=(1n<<64n)-1n;
                let state=0x123456789abcdef0n;
                function next(){
                    state=(state+0x9e3779b97f4a7c15n)&mask;
                    let x=state;
                    x=((x^(x>>30n))*0xbf58476d1ce4e5b9n)&mask;
                    x=((x^(x>>27n))*0x94d049bb133111ebn)&mask;
                    return(x^(x>>31n))&mask;
                }
                function supplement(word,index){
                    let x=(word+BigInt(index)*0x9e3779b97f4a7c15n)&mask;
                    x=((x^(x>>30n))*0xbf58476d1ce4e5b9n)&mask;
                    x=((x^(x>>27n))*0x94d049bb133111ebn)&mask;
                    return(x^(x>>31n))&mask;
                }
                async function sample(module,word){
                    const iterator=module.sampleKorea(word);
                    let response=await iterator.next(),reads=0;
                    while(!response.done&&reads<128){
                        reads++;
                        response=await iterator.next({value:supplement(word,reads),bitsRemaining:64});
                    }
                    if(!response.done)throw Error("sampler attempts exceeded");
                    return {point:response.value,reads};
                }
                let maxLatitudeDifference=0,maxLongitudeDifference=0,attemptChanges=0,regionChanges=0;
                let totalSupplementsOld=0,totalSupplementsNew=0;
                let oldMilliseconds=0,newMilliseconds=0;
                const N=160;
                const passTimings=[];
                for(let pass=0;pass<15;pass++){
                const oldStart=oldMilliseconds,newStart=newMilliseconds;
                for(let i=0;i<N;i++){
                    const word=next();
                    let oldResult,newResult;
                    if(i%2===0){
                        let t=performance.now();
                        oldResult=await sample(oldModule,word);
                        oldMilliseconds+=performance.now()-t;
                        t=performance.now();
                        newResult=await sample(newModule,word);
                        newMilliseconds+=performance.now()-t;
                    }else{
                        let t=performance.now();
                        newResult=await sample(newModule,word);
                        newMilliseconds+=performance.now()-t;
                        t=performance.now();
                        oldResult=await sample(oldModule,word);
                        oldMilliseconds+=performance.now()-t;
                    }
                    totalSupplementsOld+=oldResult.reads;
                    totalSupplementsNew+=newResult.reads;
                    if(oldResult.reads!==newResult.reads)attemptChanges++;
                    const [lat0,lon0]=oldResult.point,[lat1,lon1]=newResult.point;
                    maxLatitudeDifference=Math.max(maxLatitudeDifference,Math.abs(lat0-lat1));
                    maxLongitudeDifference=Math.max(maxLongitudeDifference,Math.abs(lon0-lon1));
                    if(Math.floor(lat0*1000)!==Math.floor(lat1*1000)||Math.floor(lon0*1000)!==Math.floor(lon1*1000))regionChanges++;
                    if(!(lat1>=33.0&&lat1<39&&lon1>=124&&lon1<132))throw Error("coordinate escaped bounds");
                }
                passTimings.push({pass,oldMs:oldMilliseconds-oldStart,newMs:newMilliseconds-newStart});
                }
                return {N,passTimings,maxLatitudeDifference,maxLongitudeDifference,attemptChanges,regionChanges,totalSupplementsOld,totalSupplementsNew,oldMilliseconds,newMilliseconds};
            },undefined,{timeout:240000});
            const summary={browser:engineName,version:browser.version(),...result};
            console.log(JSON.stringify(summary));
            results.push(summary);
            if(result.attemptChanges!==0||result.regionChanges!==0||result.maxLatitudeDifference!==0||result.maxLongitudeDifference!==0)throw Error("real boundary equivalence failed");
        }finally{await browser.close();}
    }
}finally{await new Promise(done=>server.close(done));}
console.log(JSON.stringify({passed:true,engines:results.length}));
