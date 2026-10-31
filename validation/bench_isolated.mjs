import {readFileSync} from "node:fs";
import {execFileSync} from "node:child_process";
import {chromium,firefox,webkit} from "playwright";
const moduleText=execFileSync("git",["show","d749b386c7a0aad665cb8686ddbd78abcc894cb8:korea_boundary.js"],{encoding:"utf8"});
const coreText=moduleText.slice(moduleText.indexOf("const DEGREES ="),moduleText.indexOf("const MAX_U64 ="));
const bootstrap=moduleText.match(/Uint8Array\.fromBase64\("([A-Za-z0-9+/=]+)"\)/)[1];
const compact=Buffer.from(bootstrap,"base64");
const tileMask=compact.subarray(compact.length-32768);
const tileIndex=[];
for(let tile=0;tile<262144;tile++)if(tileMask[tile>>>3]&(1<<(tile&7)))tileIndex.push(tile);
if(tileIndex.length!==79113)throw Error("tile index count wrong");
for(const [engineName,engine] of [["chromium",chromium],["firefox",firefox],["webkit",webkit]]){
    const browser=await engine.launch({headless:true});
    try{
        const page=await browser.newPage();
        const output=await page.evaluate(({coreText,tileIndex})=>{
            const mask=(1n<<64n)-1n;
            let seed=0x123456789abcdef0n;
            function next(){
                seed=(seed+0x9e3779b97f4a7c15n)&mask;
                let x=seed;
                x=((x^(x>>30n))*0xbf58476d1ce4e5b9n)&mask;
                x=((x^(x>>27n))*0x94d049bb133111ebn)&mask;
                return(x^(x>>31n))&mask;
            }
            const values=Array.from({length:90000},next);
            const candidate=Function(coreText+"\nreturn candidate;")();
            const cachedCore=coreText.replace("function candidate(value, bounds) {","function candidate(value, precomputed) {\n    const bounds = precomputed.bounds;")
                .replace("const southArea = cumulativeArea(south);","const southArea = precomputed.southArea;")
                .replace("cumulativeArea(north) - southArea","precomputed.northArea - southArea");
            const cachedCandidate=Function(cachedCore+"\nreturn candidate;")();
            const cumulativeArea=Function(coreText+"\nreturn cumulativeArea;")();
            const bounds=[124.60971767856728,33.11235575873201,131.87278314710602,38.61370930167485];
            const precomputed={bounds,southArea:cumulativeArea(bounds[1]),northArea:cumulativeArea(bounds[3])};
            const words=values.map(value=>[Number(value>>32n),Number(value&0xffffffffn)]);
            function median(data){const sorted=data.slice().sort((a,b)=>a-b);return sorted[Math.floor(sorted.length/2)];}
            function bench(fn){
                let sink=0;
                for(let warm=0;warm<2;warm++)sink+=fn();
                const samples=[];
                for(let repeat=0;repeat<9;repeat++){
                    const now=performance.now();
                    sink+=fn();
                    samples.push(performance.now()-now);
                }
                return {medianMs:median(samples),minMs:Math.min(...samples),maxMs:Math.max(...samples),sink};
            }
            let areaMismatch=0;
            for(let i=0;i<words.length;i++){
                const a=candidate(words[i],bounds),b=cachedCandidate(words[i],precomputed);
                if(!Object.is(a[0],b[0])||!Object.is(a[1],b[1]))areaMismatch++;
            }
            const baseArea=bench(()=>{let v=0;for(const value of words)v+=candidate(value,bounds)[0];return v;});
            const cachedArea=bench(()=>{let v=0;for(const value of words)v+=cachedCandidate(value,precomputed)[0];return v;});
            const dataView=new DataView(new ArrayBuffer(8));
            const tiles=Uint32Array.from(tileIndex);
            function remapOld(value){
                const selector=Number(value>>46n);
                if(selector>=79113*3)return null;
                const tile=tiles[Math.floor(selector/3)];
                return [Math.floor(tile/512)*8388608+Number((value>>23n)&0x7fffffn),(tile%512)*8388608+Number(value&0x7fffffn)];
            }
            function remapDataView(value){
                dataView.setBigUint64(0,value,false);
                const hi=dataView.getUint32(0,false),lo=dataView.getUint32(4,false);
                const selector=hi>>>14;
                if(selector>=79113*3)return null;
                const tile=tiles[Math.floor(selector/3)];
                return [Math.floor(tile/512)*8388608+((hi&0x3fff)*512+(lo>>>23)),(tile%512)*8388608+(lo&0x7fffff)];
            }
            function remap22(value){
                const selector=Number(value>>42n);
                if(selector>=79113*53)return null;
                const tile=tiles[Math.floor(selector/53)];
                const vertical=Math.floor(tile/512)*8388608+Number((value>>21n)&0x1fffffn)*4+2;
                const horizontal=(tile%512)*8388608+Number(value&0x1fffffn)*4+2;
                return [vertical,horizontal];
            }
            let remapErrors=0,oldRejected=0,newRejected=0;
            for(let i=0;i<values.length;i++){
                const a=remapOld(values[i]),b=remapDataView(values[i]),c=remap22(values[i]);
                if(JSON.stringify(a)!==JSON.stringify(b))remapErrors++;
                if(a===null)oldRejected++;
                if(c===null)newRejected++;
            }
            const oldRemap=bench(()=>{let v=0;for(const x of values){const o=remapOld(x);if(o)v+=o[0]+o[1];}return v;});
            const viewRemap=bench(()=>{let v=0;for(const x of values){const o=remapDataView(x);if(o)v+=o[0]+o[1];}return v;});
            const new22Remap=bench(()=>{let v=0;for(const x of values){const o=remap22(x);if(o)v+=o[0]+o[1];}return v;});
            const mixed=Uint16Array.from(Array.from({length:2577},(_,i)=>Math.floor(i*41240/2576)));
            const direct=new Uint16Array(41241).fill(65535);
            for(let i=0;i<mixed.length;i++)direct[mixed[i]]=i;
            function binaryOrdinal(tile){let low=0,high=mixed.length;while(low<high){const mid=(low+high)>>>1;if(mixed[mid]<tile)low=mid+1;else high=mid;}if(mixed[low]!==tile)throw Error("wrong mixed tile");return low;}
            function directOrdinal(tile){const ordinal=direct[tile];if(ordinal===65535)throw Error("wrong mixed tile");return ordinal;}
            let mixedMismatch=0;
            for(const tile of mixed)if(binaryOrdinal(tile)!==directOrdinal(tile))mixedMismatch++;
            const tileTests=Array.from({length:90000},(_,i)=>mixed[(i*977)%mixed.length]);
            const binaryAtlas=bench(()=>{let s=0;for(const v of tileTests)s+=binaryOrdinal(v);return s;});
            const directAtlas=bench(()=>{let s=0;for(const v of tileTests)s+=directOrdinal(v);return s;});
            const count=25348,partCount=96;
            const index=new DataView(new ArrayBuffer(partCount*48));
            for(let i=0;i<partCount;i++){const begin=Math.floor(i*count/partCount),end=Math.floor((i+1)*count/partCount);index.setUint32(i*48,begin,true);index.setUint32(i*48+4,end-begin,true);}
            function partBinary(page){let lo=0,hi=partCount;while(lo<hi){const m=Math.floor((lo+hi)/2);const end=index.getUint32(m*48,true)+index.getUint32(m*48+4,true);if(end<=page)lo=m+1;else hi=m;}return lo;}
            const lookup=new Uint8Array(count);for(let i=0;i<partCount;i++){const start=index.getUint32(i*48,true),end=start+index.getUint32(i*48+4,true);lookup.fill(i,start,end);}
            function partDirect(page){return lookup[page];}
            let partMismatch=0;for(let i=0;i<count;i++)if(partBinary(i)!==partDirect(i))partMismatch++;
            const pages=Array.from({length:90000},(_,i)=>Number(values[i]%BigInt(count)));
            const partOld=bench(()=>{let s=0;for(const p of pages)s+=partBinary(p);return s;});
            const partNew=bench(()=>{let s=0;for(const p of pages)s+=partDirect(p);return s;});
            return {samples:values.length,area:{mismatches:areaMismatch,baseline:baseArea,candidate:cachedArea,ratio:cachedArea.medianMs/baseArea.medianMs},
              remap:{mismatches:remapErrors,baseline:oldRemap,dataview:viewRemap,ratio:viewRemap.medianMs/oldRemap.medianMs},
              mapping22:{rejectedBefore:oldRejected,rejectedAfter:newRejected,rejectOld:oldRejected/values.length,rejectNew:newRejected/values.length,candidate:new22Remap,ratio:new22Remap.medianMs/oldRemap.medianMs},
              atlas:{mismatches:mixedMismatch,baseline:binaryAtlas,candidate:directAtlas,ratio:directAtlas.medianMs/binaryAtlas.medianMs,memoryExtraBytes:direct.byteLength},
              partIndex:{mismatches:partMismatch,baseline:partOld,candidate:partNew,ratio:partNew.medianMs/partOld.medianMs,memoryExtraBytes:lookup.byteLength}};
        },{coreText,tileIndex});
        console.log("ISOLATED "+JSON.stringify({engine:engineName,version:browser.version(),...output}));
        if(output.area.mismatches||output.remap.mismatches||output.atlas.mismatches||output.partIndex.mismatches)throw Error("precision mismatch");
    }finally{await browser.close();}
}
