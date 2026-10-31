import {readFileSync,writeFileSync,rmSync} from "node:fs";
import {resolve} from "node:path";
import {pathToFileURL,fileURLToPath} from "node:url";
import {createHash} from "node:crypto";

if(typeof Uint8Array.fromBase64!=="function")
    Uint8Array.fromBase64=text=>Uint8Array.from(Buffer.from(text,"base64"));
const source=readFileSync("korea_boundary.js","utf8");
const temp=resolve("validation/geo_boundary_private.mjs");
const originalFetch=globalThis.fetch;
writeFileSync(temp,source+"\nexport {loadBoundary,decodeAtlas,tilePages,covers};\n");
globalThis.fetch=async url=>{
    const file=fileURLToPath(url);
    const bytes=readFileSync(file);
    return new Response(bytes,{status:200,headers:{"Content-Type":"application/octet-stream"}});
};
let stats;
try{
    const {loadBoundary,decodeAtlas,tilePages,covers}=await import(pathToFileURL(temp).href+"?proof=1");
    const boundary=await loadBoundary();
    const atlas=decodeAtlas(new Uint8Array(readFileSync("data/korea-boundary-v4/atlas-8.bin")),boundary);
    const fine=32;
    const ordinals=atlas.mixedOrdinal;
    const N=ordinals.reduce((total,x)=>total+(x!==65535?1:0),0);
    const memory32=new Uint8Array(96+N*fine*fine/4);
    const header=readFileSync("data/korea-boundary-v4/atlas-8.bin").subarray(0,96);
    memory32.set(header);
    const view32=new DataView(memory32.buffer);
    view32.setUint32(12,32,true);
    view32.setUint32(88,memory32.length-96,true);
    const oldStates=atlas.states;
    const write=(bytes,subdivision,ordinal,row,col,value)=>{
        const index=(ordinal*subdivision+row)*subdivision+col;
        bytes[96+(index>>>2)]|=value<<((index&3)*2);
    };
    const readOld=(ordinal,row,col)=>{
        const index=(ordinal*8+row)*8+col;
        return (oldStates[index>>>2]>>>((index&3)*2))&3;
    };
    const markCells=(marks,baseX,baseY,step,aX,aY,bX,bY)=>{
        const minCol=Math.max(0,Math.floor((Math.min(aX,bX)-baseX)/step-1e-9));
        const maxCol=Math.min(31,Math.floor((Math.max(aX,bX)-baseX)/step+1e-9));
        const minRow=Math.max(0,Math.floor((Math.min(aY,bY)-baseY)/step-1e-9));
        const maxRow=Math.min(31,Math.floor((Math.max(aY,bY)-baseY)/step+1e-9));
        for(let row=minRow;row<=maxRow;row++)
            for(let col=minCol;col<=maxCol;col++)marks[row*32+col]=1;
    };
    const lowerBound=(items,target)=>{
        let lo=0,hi=items.length;
        while(lo<hi){const mid=(lo+hi)>>>1;if(items[mid]<=target)lo=mid+1;else hi=mid;}
        return lo;
    };
    const count=[0,0,0];
    let mixedFrom8=0,refinedInside=0,refinedOutside=0,checked=0,issues=0,tiles=0,ringEdges=0,parityEvents=0,pageVisits=0;
    const sampleCells=[];
    for(let tile=0;tile<boundary.tileCount;tile++){
        const ordinal=ordinals[tile];
        if(ordinal===65535)continue;
        tiles++;
        const row=Math.floor(tile/boundary.columns),col=tile%boundary.columns;
        const baseX=boundary.gridWest+col/boundary.scale;
        const baseY=boundary.gridSouth+row/boundary.scale;
        const step=1/(boundary.scale*32);
        const marked=new Uint8Array(1024);
        const crossings=Array.from({length:32},()=>[]);
        const events=[];
        for await(const page of tilePages(boundary,tile)){
            pageVisits++;
            if(page.kind===0){
                events.push(page);
                for(let e=0;e<page.count;e++){
                    const latitude=page.data.getFloat64(32+e*8,true);
                    parityEvents++;
                    const r=Math.floor((latitude-baseY)/step);
                    for(let rr=Math.max(0,r-1);rr<=Math.min(31,r+1);rr++)
                        for(let cc=0;cc<32;cc++)marked[rr*32+cc]=1;
                }
                continue;
            }
            let previousX=page.data.getFloat64(40,true);
            let previousY=page.data.getFloat64(48,true);
            for(let v=1;v<=page.count;v++){
                const longitude=page.data.getFloat64(40+v*16,true);
                const latitude=page.data.getFloat64(48+v*16,true);
                ringEdges++;
                markCells(marked,baseX,baseY,step,previousX,previousY,longitude,latitude);
                const min=Math.max(0,Math.ceil((Math.min(previousY,latitude)-baseY)/step-0.5));
                const max=Math.min(31,Math.floor((Math.max(previousY,latitude)-baseY)/step-0.5));
                if(previousY!==latitude){
                    for(let rr=min;rr<=max;rr++){
                        const cy=baseY+(rr+0.5)*step;
                        if((previousY>cy)!==(latitude>cy))
                            crossings[rr].push(previousX+(cy-previousY)*(longitude-previousX)/(latitude-previousY));
                    }
                }
                previousX=longitude;
                previousY=latitude;
            }
        }
        for(const rowCrossings of crossings)rowCrossings.sort((a,b)=>a-b);
        for(let r=0;r<32;r++){
            const cy=baseY+(r+0.5)*step;
            let eventParity=0;
            for(const event of events){
                let lo=0,hi=event.count;
                while(lo<hi){
                    const mid=(lo+hi)>>>1;
                    if(event.data.getFloat64(32+mid*8,true)<=cy)lo=mid+1;
                    else hi=mid;
                }
                eventParity^=(event.base^(lo&1));
            }
            for(let col=0;col<32;col++){
                const x=baseX+(col+0.5)*step;
                const old=readOld(ordinal,r>>>2,col>>>2);
                let state=old;
                if(old===2){
                    mixedFrom8++;
                    if(marked[r*32+col]||x<boundary.bounds[0]||x>boundary.bounds[2]
                        ||cy<boundary.bounds[1]||cy>boundary.bounds[3])state=2;
                    else{
                        const rowCrossings=crossings[r];
                        const countToRight=rowCrossings.length-lowerBound(rowCrossings,x);
                        state=(eventParity^(countToRight&1))?1:0;
                        if(state===1)refinedInside++;
                        else refinedOutside++;
                        if((r*31+col*17+tile)%521===0)sampleCells.push({lat:cy,lon:x,state});
                    }
                }
                count[state]++;
                write(memory32,32,ordinal,r,col,state);
            }
        }
        if(tiles%500===0)console.log("ATLAS_PROGRESS "+JSON.stringify({tiles,partCount:boundary.partCount,refinedInside,refinedOutside,remainingMixed:count[2]}));
    }
    for(const cell of sampleCells){
        const actual=await covers(boundary,[cell.lat,cell.lon]);
        checked++;
        if(actual!==(cell.state===1))issues++;
    }
    if(issues)throw Error("Finer atlas certified-cell parity mismatch "+issues+"/"+checked);
    const fine16=new Uint8Array(96+N*16*16/4);
    fine16.set(header);
    const view16=new DataView(fine16.buffer);
    view16.setUint32(12,16,true);
    view16.setUint32(88,fine16.length-96,true);
    const count16=[0,0,0];
    const read32=(ordinal,r,c)=>{
        const index=(ordinal*32+r)*32+c;
        return (memory32[96+(index>>>2)]>>>((index&3)*2))&3;
    };
    for(let ordinal=0;ordinal<N;ordinal++)
        for(let r=0;r<16;r++)
            for(let col=0;col<16;col++){
                const a=read32(ordinal,r*2,col*2);
                const b=read32(ordinal,r*2,col*2+1);
                const c=read32(ordinal,r*2+1,col*2);
                const d=read32(ordinal,r*2+1,col*2+1);
                const state=a===b&&a===c&&a===d?a:2;
                write(fine16,16,ordinal,r,col,state);
                count16[state]++;
            }
    writeFileSync("validation/geo_atlas16.bin",fine16);
    writeFileSync("validation/geo_atlas32.bin",memory32);
    stats={
        mixedTiles:N,tiles,original8Bytes:header.length+N*8*8/4,
        level16:{bytes:fine16.length,states:count16,mixedFraction:count16[2]/count16.reduce((a,b)=>a+b,0),sha256:createHash("sha256").update(fine16).digest("hex")},
        level32:{bytes:memory32.length,states:count,mixedFraction:count[2]/count.reduce((a,b)=>a+b,0),sha256:createHash("sha256").update(memory32).digest("hex")},
        mixedCellsRefined:refinedInside+refinedOutside,originalMixedFineCells:mixedFrom8,
        checkedCertifiedCells:checked,certificationMismatches:issues,ringEdges,parityEvents,pageVisits
    };
    console.log("ATLAS_REFINED "+JSON.stringify(stats));
}finally{
    globalThis.fetch=originalFetch;
    rmSync(temp,{force:true});
}
