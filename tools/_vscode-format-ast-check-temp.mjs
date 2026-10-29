import fs from "node:fs";
import crypto from "node:crypto";
import ts from "typescript";

const before=fs.readFileSync("revgeo.before-format.html","utf8");
const after=fs.readFileSync("revgeo.html","utf8");
const expectedBefore="8fa68e67cb4dbf54a22cdf7af16350ccc95ebbff9072ba2528a07f96e8679aee";
const expectedAfter="bfd03ebc06cda0b76f6c6a294c9ba9ddce322833ab1e9d2e1174f5a2b7ca68cd";
const hash=s=>crypto.createHash("sha256").update(s).digest("hex");
if(hash(before)!==expectedBefore)throw Error("Wrong baseline HTML");
if(hash(after)!==expectedAfter)throw Error("VS Code output differs from verified formatter run");
function capture(pat,s){
 const m=s.match(pat);
 if(!m)throw Error("Expected HTML tag missing");
 return m[1];
}
const jsRe=/<script\b[^>]*>([\s\S]*?)<\/script>/i;
const styleRe=/<style\b[^>]*>([\s\S]*?)<\/style>/i;
if(capture(styleRe,before)!==capture(styleRe,after))throw Error("CSS changed");
for(const tag of ["script","style","textarea","pre"]){
 const re=new RegExp("<"+tag+"\\b","gi");
 if([...before.matchAll(re)].length!==[...after.matchAll(re)].length)throw Error(tag+" count changed");
}
function syntax(source){
 const ast=ts.createSourceFile("script.mjs",source,ts.ScriptTarget.ESNext,true,ts.ScriptKind.JS);
 if(ast.parseDiagnostics.length)throw Error("JS parse diagnostics: "+ast.parseDiagnostics.map(x=>x.messageText));
 const nodes=[];
 function visit(n){
  nodes.push([n.kind,n.kind!==ts.SyntaxKind.SourceFile&&typeof n.text==="string"?n.text:"",typeof n.escapedText==="string"?n.escapedText:"",typeof n.rawText==="string"?n.rawText:""]);
  ts.forEachChild(n,visit);
 }
 visit(ast);
 return nodes;
}
const a=syntax(capture(jsRe,before)),b=syntax(capture(jsRe,after));
if(JSON.stringify(a)!==JSON.stringify(b))throw Error("JavaScript AST changed");
const results={sourceSha256:hash(before),formattedSha256:hash(after),sameJavascriptAST:true,javascriptSyntaxNodes:a.length,identicalStyle:true,scriptAndStructureTagCountUnchanged:true,baselineBytes:Buffer.byteLength(before),formattedBytes:Buffer.byteLength(after),lineCountBaseline:before.split("\n").length,lineCountFormatted:after.split("\n").length};
fs.mkdirSync("vscode-format-evidence",{recursive:true});
fs.writeFileSync("vscode-format-evidence/semantic-check.json",JSON.stringify(results,null,2)+"\n");
console.log(JSON.stringify(results));
