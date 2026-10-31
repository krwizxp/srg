import {execFileSync,spawnSync} from "node:child_process";
import {mkdtempSync,readFileSync,writeFileSync,rmSync} from "node:fs";
import {tmpdir} from "node:os";
import {join} from "node:path";

const reference="57ddaa4f63584797ae8d22a64fb6c01149ae59e6";
const candidates=[
    "CSS: “field-sizing”: Property “field-sizing” doesn't exist.",
    "CSS: “anchor-name”: Property “anchor-name” doesn't exist.",
    "CSS: “anchor-name”: Property “anchor-name” doesn't exist.",
    "CSS: Unrecognized at-rule “@position-try”",
    "CSS: “position-try-fallbacks”: Property “position-try-fallbacks” doesn't exist.",
    "CSS: “position-anchor”: Property “position-anchor” doesn't exist.",
    "CSS: “position-anchor”: Property “position-anchor” doesn't exist.",
    "CSS: “position-area”: Property “position-area” doesn't exist.",
    "CSS: “position-area”: Property “position-area” doesn't exist."
].sort();
const temp=mkdtempSync(join(tmpdir(),"srg-nu-reference-"));
try{
    const baseline=join(temp,"baseline-revgeo.html");
    writeFileSync(baseline,execFileSync("git",["show",reference+":revgeo.html"]));
    const jar=execFileSync("find",["validation/node_modules/vnu-jar","-name","vnu.jar","-print","-quit"],{encoding:"utf8"}).trim();
    if(!jar)throw Error("Nu Html Checker vnu.jar is missing");
    const check=filename=>{
        const result=spawnSync("java",["-jar",jar,"--errors-only",filename],{
            encoding:"utf8",maxBuffer:4*1024*1024
        });
        if(result.error)throw result.error;
        const combined=(result.stdout||"")+"\n"+(result.stderr||"");
        const lines=combined.split(/\r?\n/).filter(Boolean);
        const messages=lines.map(line=>{
            const match=line.match(/:\s*error:\s*(.*)$/);
            return match?.[1]??null;
        });
        const unrelated=lines.filter((line,i)=>messages[i]===null);
        if(unrelated.length)throw Error("Unexpected checker output: "+unrelated.slice(0,5).join("; "));
        if(result.status!==0 && result.status!==1)
            throw Error("Nu check terminated unexpectedly: "+result.status);
        if(result.status===0 && messages.length)
            throw Error("Nu check reported errors but exited successfully");
        return messages.sort();
    };
    const original=check(baseline);
    const modified=check("revgeo.html");
    const match=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
    console.log("NU_CSS_EXCEPTION "+JSON.stringify({
        baseline:reference,
        originalErrors:original.length,
        candidateErrors:modified.length,
        ignoredOnlyKnownCSS:match(original,candidates)&&match(modified,candidates),
        baselineMatchesCandidate:match(original,modified),
        ignoredDiagnostics:modified,
        otherErrors:0
    }));
    if(!match(original,modified)||!(match(original,candidates)||(original.length===0&&modified.length===0)))
        throw Error("The current HTML introduced a new Nu error or the errors are not the approved 9 CSS syntax diagnostics");
    if(!readFileSync("revgeo.html","utf8").includes('import { convertParallel } from "./srg_parallel.js?v=1";'))
        throw Error("Worker integration is missing from final saved HTML");
}finally{
    rmSync(temp,{recursive:true,force:true});
}
