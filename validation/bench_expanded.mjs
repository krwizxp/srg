import {readFileSync} from "node:fs";
import {chromium,firefox,webkit} from "playwright";
const moduleText=readFileSync("korea_boundary.js","utf8");
const inflateSource=moduleText.slice(0,moduleText.indexOf("const DEGREES ="));
const orientationSource=moduleText.slice(moduleText.indexOf("function productError("),moduleText.indexOf("function cumulativeArea("));
const pages=[];
for(const number of [0,12,40,80,95]){
    const name="data/korea-boundary-v4/part-"+String(number).padStart(3,"0")+".bin";
    const data=readFileSync(name);
    const view=new DataView(data.buffer,data.byteOffset,data.byteLength);
    const count=view.getUint32(12,true);
    for(let index=0;index<Math.min(count,15);index++){
        const entry=64+index*12;
        const start=view.getUint32(entry,true),length=view.getUint32(entry+4,true),rawLength=view.getUint32(entry+8,true);
        if(rawLength>32768||start+length>data.length)throw Error("bad part entry");
        pages.push({input:data.subarray(start,start+length).toString("base64"),length:rawLength});
    }
}
for(const [name,engine] of [["chromium",chromium],["firefox",firefox],["webkit",webkit]]){
    const browser=await engine.launch({headless:true});
    try{
        const page=await browser.newPage();
        const r=await page.evaluate(async ({inflateSource,orientationSource,pages})=>{
            const inflate=Function(inflateSource+"\nreturn inflate;")();
            const orientation=Function(orientationSource+"\nreturn orientation;")();
            const inputs=pages.map(p=>({bytes:Uint8Array.fromBase64(p.input),length:p.length}));
            async function native({bytes}) {
                return new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate"))).arrayBuffer());
            }
            let mismatches=0;
            let totalBytes=0;
            for(const p of inputs){
                const a=inflate(p.bytes,p.length),b=await native(p);
                totalBytes+=a.length;
                if(a.length!==b.length||a.some((value,i)=>value!==b[i]))mismatches++;
            }
            let checks=0;
            const now=()=>performance.now();
            async function timeAsync(fn){
                for(let i=0;i<inputs.length;i++)await fn(inputs[i]);
                const samples=[];
                let sink=0;
                for(let rep=0;rep<5;rep++){
                    const t=now();
                    for(const p of inputs) sink+=(await fn(p)).length;
                    samples.push(now()-t);
                }
                samples.sort((a,b)=>a-b);
                return {medianMs:samples[2],minMs:samples[0],maxMs:samples[4],sink};
            }
            const jsDecode=await timeAsync(p=>Promise.resolve(inflate(p.bytes,p.length)));
            const nativeDecode=await timeAsync(native);
            const ring=new Float64Array(2050);
            for(let i=0;i<=1024;i++){
                ring[2*i]=127+0.8*Math.sin(i*0.138)+0.2*Math.sin(i*1.1);
                ring[2*i+1]=35+0.6*Math.sin(i*0.097);
            }
            function oldEdge(location){
                let start=[ring[0],ring[1]];
                let parity=false;
                for(let i=1;i<=1024;i++){
                    const end=[ring[i*2],ring[i*2+1]];
                    const latitude=location[1];
                    if(latitude>=Math.min(start[1],end[1])&&latitude<=Math.max(start[1],end[1])){
                        const side=orientation(start,end,location);
                        if((start[1]>latitude)!==(end[1]>latitude)){
                            const crossing=end[1]>start[1]?side>0:side<0;
                            parity=parity!==crossing;
                        }
                    }
                    start=end;
                }
                return parity;
            }
            function reuseEdge(location){
                const start=[ring[0],ring[1]],end=[0,0];
                let parity=false;
                for(let i=1;i<=1024;i++){
                    end[0]=ring[i*2];
                    end[1]=ring[i*2+1];
                    const latitude=location[1];
                    if(latitude>=Math.min(start[1],end[1])&&latitude<=Math.max(start[1],end[1])){
                        const side=orientation(start,end,location);
                        if((start[1]>latitude)!==(end[1]>latitude)){
                            const crossing=end[1]>start[1]?side>0:side<0;
                            parity=parity!==crossing;
                        }
                    }
                    start[0]=end[0];
                    start[1]=end[1];
                }
                return parity;
            }
            const points=Array.from({length:1500},(_,i)=>[127+(i%79)/99,35+(i%37)/59]);
            let edgeMismatch=0;
            for(const point of points)if(oldEdge(point)!==reuseEdge(point))edgeMismatch++;
            function timeEdges(fn){
                const samples=[];let sink=0;
                for(let warm=0;warm<2;warm++)for(const p of points)sink+=fn(p);
                for(let rep=0;rep<7;rep++){
                    const t=now();
                    for(const p of points)sink+=fn(p);
                    samples.push(now()-t);
                }
                samples.sort((a,b)=>a-b);
                return {medianMs:samples[3],minMs:samples[0],maxMs:samples[6],sink};
            }
            const originalEdges=timeEdges(oldEdge);
            const reusedEdges=timeEdges(reuseEdge);
            return {pages:inputs.length,totalBytes,inflate:{mismatches,baseline:jsDecode,candidate:nativeDecode,ratio:nativeDecode.medianMs/jsDecode.medianMs},
                edge:{points:points.length,mismatches:edgeMismatch,baseline:originalEdges,candidate:reusedEdges,ratio:reusedEdges.medianMs/originalEdges.medianMs}};
        },{inflateSource,orientationSource,pages});
        console.log("EXPANDED "+JSON.stringify({engine:name,version:browser.version(),...r}));
        if(r.inflate.mismatches||r.edge.mismatches)throw Error("candidate mismatch");
    }finally{await browser.close();}
}
