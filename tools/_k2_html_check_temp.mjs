import fs from "node:fs/promises";
import path from "node:path";
import {createHash} from "node:crypto";

const root=process.cwd();
const evidence=path.join(root,"validation-evidence");
const htmlPath=path.join(root,"revgeo.html");
const raw=await fs.readFile(htmlPath);
const hash=createHash("sha256").update(raw).digest("hex");
if(hash!=="8fa68e67cb4dbf54a22cdf7af16350ccc95ebbff9072ba2528a07f96e8679aee")
 throw new Error("HTML changed from pinned main deployment");
const read=(name)=>fs.readFile(path.join(evidence,name),"utf8");
const nuStatus=Number((await read("nu.exit")).trim());
const htmlStatus=Number((await read("htmlvalidate.exit")).trim());
const nuRaw=(await read("nu.stdout"))+"\n"+(await read("nu.stderr"));
const htmlRaw=(await read("htmlvalidate.stdout"))+"\n"+(await read("htmlvalidate.stderr"));
const allowed=[
 "CSS: “field-sizing”: Property “field-sizing” doesn't exist.",
 "CSS: “anchor-name”: Property “anchor-name” doesn't exist.",
 "CSS: “anchor-name”: Property “anchor-name” doesn't exist.",
 "CSS: Unrecognized at-rule “@position-try”",
 "CSS: “position-try-fallbacks”: Property “position-try-fallbacks” doesn't exist.",
 "CSS: “position-anchor”: Property “position-anchor” doesn't exist.",
 "CSS: “position-area”: Property “position-area” doesn't exist.",
 "CSS: “position-anchor”: Property “position-anchor” doesn't exist.",
 "CSS: “position-area”: Property “position-area” doesn't exist."
];
let nu;
try{nu=JSON.parse(nuRaw.trim())}catch(e){nu={parse_error:String(e),raw:nuRaw.slice(0,2000)}}
const messages=(nu.messages||[]).map(x=>({type:x.type,message:x.message,line:x.lastLine}));
const limits=new Map();
for(const msg of allowed)limits.set(msg,(limits.get(msg)||0)+1);
const unexpected=[];
for(const entry of messages){
 if(entry.type!=="error" || !limits.get(entry.message))unexpected.push(entry);
 else limits.set(entry.message,limits.get(entry.message)-1);
}
const accepted=messages.length>0 && unexpected.length===0 && messages.length<=9 && nuStatus===1;
const clean=nuStatus===0 && messages.length===0;
const pass=htmlStatus===0 && (accepted||clean);
const report={
 sourceRevision:"beb3a69d1cdcf7e23c8e64ce10f70e446027f20c",
 sourceSHA256:hash,
 htmlvalidate:{exitCode:htmlStatus,rawOutput:htmlRaw.slice(0,9000),passed:htmlStatus===0},
 nu:{exitCode:nuStatus,messages,diagnosticCount:messages.length,unexpected,rawResult:nu.parse_error?"malformed_output":"parsed",acceptedHistoricalException:accepted,clean,passedUnderExistingPolicy:accepted||clean},
 htmlUnmodified:true,
 finalPass:pass
};
await fs.writeFile(path.join(evidence,"html-validator-results.json"),JSON.stringify(report,null,2));
console.log(JSON.stringify(report));
if(!pass)process.exitCode=1;
