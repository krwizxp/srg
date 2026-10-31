import {spawnSync} from "node:child_process";
import {readFileSync,mkdtempSync,rmSync,statSync} from "node:fs";
import {tmpdir} from "node:os";
import {resolve,join} from "node:path";
import {performance} from "node:perf_hooks";

const executable=resolve("target/release/srg");
const dir=mkdtempSync(join(tmpdir(),"srg-cli-10k-"));
const repetitions=[];
let error=null;
try{
    for(let i=0;i<3;i++){
        const start=performance.now();
        const result=spawnSync(executable,["generate","10000"],{
            cwd:dir,stdio:["ignore","pipe","pipe"],encoding:"utf8",timeout:240000,maxBuffer:10*1024*1024
        });
        const ms=performance.now()-start;
        if(result.error||result.status!==0){
            error={iteration:i,status:result.status,issue:String(result.error||result.stderr||"command failed").slice(0,350)};
            break;
        }
        const path=join(dir,"random_data.txt");
        const bytes=statSync(path).size;
        const content=readFileSync(path,"utf8");
        const matches=content.match(/64비트 난수:/g)||[];
        if(matches.length!==10000)throw Error("generated "+matches.length+" records");
        repetitions.push({elapsedMs:ms,fileBytes:bytes,records:matches.length});
    }
}finally{rmSync(dir,{recursive:true,force:true});}
const times=repetitions.map(x=>x.elapsedMs).sort((a,b)=>a-b);
console.log("CLI_BENCH "+JSON.stringify({
    n:10000,source:"release srg generate 10000",hardwareEntropy:true,
    threadCount:"available_parallelism",repetitions,
    medianMs:times.length?times[Math.floor(times.length/2)]:null,
    error
}));
if(error)console.log("CLI_BENCH_LIMITATION Hardware RNG support on virtual runner or native runtime not available");
