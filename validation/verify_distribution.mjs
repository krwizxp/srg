import {readFileSync} from "node:fs";

const baseline=readFileSync("validation/baseline-distribution.txt","utf8").trim().split(/\r?\n/);
const candidate=readFileSync("validation/candidate-distribution.txt","utf8").trim().split(/\r?\n/);
if(baseline.length!==candidate.length||baseline.length!==120000)throw Error("distribution sample count mismatch");
const view=new DataView(new ArrayBuffer(8));
function floatFromHex(text){view.setBigUint64(0,BigInt("0x"+text),false);return view.getFloat64(0,false);}
function parse(rows){
    const points=[];
    let supplements=0;
    const regions={jeju:0,ulleung:0,dokdo:0};
    for(const row of rows){
        const [latitudeHex,longitudeHex,readsText]=row.trim().split(" ");
        const lat=floatFromHex(latitudeHex),lon=floatFromHex(longitudeHex);
        if(!Number.isFinite(lat)||!Number.isFinite(lon)||lat<33||lat>39||lon<124||lon>132)throw Error("invalid land sample coordinate");
        supplements+=Number(readsText);
        if(lat>=33.1&&lat<=33.6&&lon>=126.1&&lon<=127.0)regions.jeju++;
        if(lat>=37.4&&lat<=37.65&&lon>=130.7&&lon<=131.0)regions.ulleung++;
        if(lat>=37.20&&lat<=37.26&&lon>=131.83&&lon<=131.89)regions.dokdo++;
        points.push([lat,lon]);
    }
    return {points,supplements,regions};
}
const old=parse(baseline),fresh=parse(candidate);
function compare(step){
    function key([lat,lon]){
        return Math.floor((lat-33)/step)+":"+Math.floor((lon-124)/step);
    }
    const a=new Map(),b=new Map();
    for(const point of old.points){const k=key(point);a.set(k,(a.get(k)||0)+1);}
    for(const point of fresh.points){const k=key(point);b.set(k,(b.get(k)||0)+1);}
    const cells=new Set([...a.keys(),...b.keys()]);
    let maxZ=0,chi=0,with100=0,with50=0;
    let mostDifferent=null;
    for(const k of cells){
        const x=a.get(k)||0,y=b.get(k)||0,total=x+y;
        if(total<10)continue;
        const z=Math.abs(x-y)/Math.sqrt(total);
        if(z>maxZ){maxZ=z;mostDifferent={key:k,before:x,after:y,z};}
        chi+=(x-y)*(x-y)/total;
        if(total>=100)with100++;
        if(total>=50)with50++;
    }
    return {step,cells:cells.size,with100,with50,maxZ,chiSquare:chi,chiOverCells:chi/cells.size,mostDifferent};
}
const grids=[0.5,0.25,0.125].map(compare);
const detail={n:baseline.length,oldSupplements:old.supplements,newSupplements:fresh.supplements,supplementRatio:fresh.supplements/old.supplements,regions:{old:old.regions,new:fresh.regions},grids};
console.log("DISTRIBUTION "+JSON.stringify(detail));
if(grids.some(g=>g.maxZ>7||g.chiOverCells>2.0))throw Error("geographic distribution statistical check failed");
