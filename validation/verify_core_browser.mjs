import {readFileSync} from "node:fs";
import {execFileSync} from "node:child_process";
import {chromium,firefox,webkit} from "playwright";
const base=execFileSync("git",["show","ee021b543e8e9cd955e8e23878afba0221d53e33:korea_boundary.js"],{encoding:"utf8"});
const now=readFileSync("korea_boundary.js","utf8");
for(const [name,engine] of [["chromium",chromium],["firefox",firefox],["webkit",webkit]]){
    const browser=await engine.launch({headless:true});
    try{
        const page=await browser.newPage();
        const result=await page.evaluate(({base,now})=>{
            const bounds=[124.60971767856728,33.11235575873201,131.87278314710602,38.61370930167485];
            function extract(source){
                const start=source.indexOf("const DEGREES ="),end=source.indexOf("const MAX_U64 =");
                if(start<0||end<start)throw Error("source not found");
                return Function(source.slice(start,end)+"\nreturn candidate")();
            }
            const old=extract(base),candidate=extract(now);
            let seed=0x123456789abcdef0n;
            const mask=(1n<<64n)-1n;
            function next(){
                seed=(seed+0x9e3779b97f4a7c15n)&mask;
                let x=seed;
                x=((x^(x>>30n))*0xbf58476d1ce4e5b9n)&mask;
                x=((x^(x>>27n))*0x94d049bb133111ebn)&mask;
                return(x^(x>>31n))&mask;
            }
            const values=Array.from({length:30000},next);
            let maxDifference=0,errors=0;
            for(const value of values){
                const upper=Number(value>>32n),lower=Number(value&0xffffffffn);
                const oldPoint=old(value,bounds),newPoint=candidate([upper,lower],bounds);
                maxDifference=Math.max(maxDifference,Math.abs(oldPoint[0]-newPoint[0]));
                if(Math.abs(oldPoint[0]-newPoint[0])>1e-12||!Object.is(oldPoint[1],newPoint[1]))errors++;
            }
            let checksum=0;
            function run(fn){
                let sum=0;
                for(const value of values)sum+=fn(value)[0];
                checksum+=sum;
            }
            const methods=[["old",v=>old(v,bounds)],["optimized",v=>candidate([Number(v>>32n),Number(v&0xffffffffn)],bounds)]];
            for(let w=0;w<3;w++)for(const [,fn] of methods)run(fn);
            const trials={old:[],optimized:[]};
            for(let repeat=0;repeat<11;repeat++){
                for(const [method,fn] of (repeat%2?methods.slice().reverse():methods)){
                    const start=performance.now();
                    run(fn);
                    trials[method].push((performance.now()-start)*1000/values.length);
                }
            }
            function median(xs){const sorted=xs.slice().sort((a,b)=>a-b);return sorted[Math.floor(sorted.length/2)];}
            return {samples:values.length,maxDifference,errors,baselineMedianUs:median(trials.old),candidateMedianUs:median(trials.optimized),ratio:median(trials.optimized)/median(trials.old),baseline:trials.old,candidate:trials.optimized,checksum};
        },{base,now});
        console.log(JSON.stringify({name,version:browser.version(),...result}));
        if(result.errors>0||result.ratio>0.6)throw Error("browser core fails equivalence or speed");
    }finally{await browser.close();}
}
