import fs from "node:fs/promises";
import {createReadStream} from "node:fs";
import http from "node:http";
import path from "node:path";
import {createHash} from "node:crypto";
import {chromium, firefox, webkit} from "playwright";

const root=process.cwd();
const output=process.env.GITHUB_WORKSPACE+"/validation-evidence";
await fs.mkdir(output,{recursive:true});
const expected={
 "revgeo.html":"bfd03ebc06cda0b76f6c6a294c9ba9ddce322833ab1e9d2e1174f5a2b7ca68cd",
 "korea_boundary.js":"f785171eaafb9d2c332a30122778d9c1ceabf92ac062c0b8f6bcca00dfede0e4",
 "data/korea-boundary-v4/tile-select-v5.bin":"492fdbeee7e7a64816d7ea90c57cb5773685fbf96e0b95911a8b7cdcfbf2ae8b"
};
for(const [name,hash] of Object.entries(expected)){
 const data=await fs.readFile(path.join(root,name));
 if(createHash("sha256").update(data).digest("hex")!==hash)throw new Error("Wrong deployed source "+name);
}
const bitset=await fs.readFile(path.join(root,"data/korea-boundary-v4/tile-select-v5.bin"));
const tiles=[];
for(let i=0;i<262144;i++)if(bitset[i>>>3]&(1<<(i&7)))tiles.push(i);
if(tiles.length!==79113)throw new Error("Incorrect tile count");
function encodedOriginal(v){
 const vertical=v>>32n,horizontal=v&0xffffffffn;
 const tile=Number(((vertical>>23n)<<9n)|(horizontal>>23n));
 const rank=tiles.indexOf(tile);
 if(rank<0)throw new Error("Reference mainland point not selected");
 return ((BigInt(rank)*2n)<<46n)|((vertical&0x7fffffn)<<23n)|(horizontal&0x7fffffn);
}
const encoded=encodedOriginal(15033626022739801206n);
const cases=[];
const failures=[];
let server;
try{
 server=http.createServer(async (req,res)=>{
  try{
   const raw=new URL(req.url,"http://127.0.0.1").pathname;
   const filename=path.resolve(root,"."+decodeURIComponent(raw));
   if(filename!==root&&!filename.startsWith(root+path.sep)){res.writeHead(403);res.end();return}
   const stat=await fs.stat(filename);
   if(!stat.isFile()){res.writeHead(404);res.end();return}
   const ext=path.extname(filename);
   const mime={".html":"text/html; charset=utf-8",".js":"text/javascript; charset=utf-8",".bin":"application/octet-stream",".css":"text/css",".svg":"image/svg+xml",".json":"application/json",".webmanifest":"application/manifest+json"}[ext]||"application/octet-stream";
   res.writeHead(200,{"content-type":mime,"content-length":stat.size,"cache-control":"no-store"});
   createReadStream(filename).pipe(res);
  }catch(e){res.writeHead(404);res.end("not found")}
 });
 await new Promise(resolve=>server.listen(18973,"127.0.0.1",resolve));
 const url="http://127.0.0.1:18973/revgeo.html";
 const fixture=[
  "대한민국 위경도: 37.56, 126.98",
  "세계 위경도: 40.7, -74.0",
  "세계 위경도: 35.7, 139.6",
  "세계 위경도: 39.02, 125.75"
 ].join("\n")+"\n";
 const datasets=[
  ["37.56","kr","대한민국","서울"],
  ["40.7","us","미국","New York, United States"],
  ["35.7","jp","일본","Tokyo, Japan"],
  ["39.02","kp","조선민주주의인민공화국","평양"]
 ];
 for(const [name,type] of [["Chromium",chromium],["Firefox",firefox],["WebKit",webkit]]){
  let browser,context;
  try{
   browser=await type.launch({headless:true});
   context=await browser.newContext({acceptDownloads:true,locale:"ko-KR",viewport:{width:1365,height:1000}});
   const page=await context.newPage();
   const pageErrors=[],consoleErrors=[],requestFailures=[];
   page.on("pageerror",e=>pageErrors.push(String(e)));
   page.on("console",m=>{if(m.type()==="error")consoleErrors.push(m.text())});
   page.on("requestfailed",req=>requestFailures.push({url:req.url(),failure:req.failure()}));
   async function check(test,fn){
    const start=performance.now();
    try{
     const detail=await fn();
     cases.push({engine:name,test,status:"passed",elapsed_ms:Math.round(performance.now()-start),detail:detail??null});
     console.log(name+" PASS "+test);
    }catch(e){
     cases.push({engine:name,test,status:"failed",elapsed_ms:Math.round(performance.now()-start),error:String(e).slice(0,3000)});
     failures.push(name+": "+test+": "+String(e));
     console.log(name+" FAIL "+test+" "+String(e));
     throw e;
    }
   }
   const wait=(script,timeout=25000)=>page.waitForFunction(script,null,{timeout});
   await check("boot_http_origin_and_default_filters",async ()=>{
    await page.goto(url,{waitUntil:"domcontentloaded",timeout:30000});
    const options=await page.locator("#filter-kind option").evaluateAll(list=>list.map(o=>[o.value,o.textContent.trim()]));
    if(JSON.stringify(options.map(x=>x[0]))!==JSON.stringify(["","kr","에러"]))throw new Error("Unexpected default kind filters "+JSON.stringify(options));
    if(options[0][1]!=="전체")throw new Error("All label not simplified");
    return {origin:await page.evaluate("location.origin"),options};
   });
   await check("real_manual_k2_conversion_and_txt_download",async ()=>{
    await page.locator("#manual-tab").click();
    await page.locator("#num64").fill(String(encoded));
    await page.locator("#auto-supp").check();
    await page.locator("#runBtn").click();
    await wait("document.querySelector('#meta').textContent.includes('완료')",60000);
    const raw=await page.locator("#output").innerText();
    if(!raw.includes("대한민국 위경도:"))throw new Error("No Korea coordinate in output");
    const downloadWait=page.waitForEvent("download",{timeout:15000});
    await page.locator("#saveOutput").click();
    const downloaded=await downloadWait;
    const text=await fs.readFile(await downloaded.path(),"utf8");
    if(!text.includes("대한민국 위경도:")||!text.includes("64비트 난수:"))throw new Error("TXT export corrupt");
    return {num64:String(encoded),output_bytes:Buffer.byteLength(text),korea_line:text.split("\n").find(x=>x.includes("대한민국 위경도:"))};
   });
   await check("use_converted_txt_as_file",async ()=>{
    await page.locator("#useOutput").click();
    await wait("document.getElementById('coords-count').textContent.trim()==='2'",18000);
    return {coordinates:await page.locator("#coords-count").innerText()};
   });
   await check("file_geocode_country_kp_filter_csv",async ()=>{
    const requests=[];
    await page.route("https://nominatim.openstreetmap.org/reverse?**",async route=>{
     const target=new URL(route.request().url());
     const lat=target.searchParams.get("lat");
     const record=datasets.find(x=>x[0]===lat);
     if(!record){await route.fulfill({status:500,contentType:"application/json",body:'{"error":"unexpected coordinate"}'});return}
     requests.push(target.toString());
     const body=JSON.stringify({features:[{properties:{geocoding:{country_code:record[1],country:record[2],label:record[3],admin:{},name:record[3]}}}]});
     await route.fulfill({status:200,contentType:"application/json; charset=utf-8",body});
    });
    await page.locator("#file-tab").click();
    await page.locator("#file").setInputFiles({name:"k2-lookup.txt",mimeType:"text/plain",buffer:Buffer.from(fixture)});
    await page.locator("#parse").click();
    await wait("document.getElementById('coords-count').textContent.trim()==='4'",18000);
    await page.locator("#start").click();
    await wait("document.getElementById('summary-processed').textContent.trim()==='4'",50000);
    await page.waitForTimeout(300);
    const options=await page.locator("#filter-kind option").evaluateAll(list=>list.map(o=>[o.value,o.textContent.trim()]));
    const values=options.map(o=>o[0]);
    for(const val of ["","kr","에러","kp","country:us","country:jp"])if(!values.includes(val))throw new Error("Missing kind "+val+" in "+JSON.stringify(options));
    const kp=options.find(x=>x[0]==="kp");if(kp?.[1]!=="북한")throw new Error("KP not labeled 북한");
    if(requests.length!==4)throw new Error("Nominatim request count "+requests.length);
    let rows=(await page.locator("#tbl tbody tr").allInnerTexts()).join("\n");
    for(const name of ["대한민국","미국","일본","북한"])if(!rows.includes(name))throw new Error("Missing row "+name);
    await page.locator("#filter-kind").selectOption("kp");
    await wait("document.querySelector('#results-count').textContent.includes('1 / 1')",8000);
    rows=(await page.locator("#tbl tbody tr").allInnerTexts()).join("\n");
    if(!rows.includes("북한")||rows.includes("미국"))throw new Error("KP filter incorrect");
    await page.locator("#filter-kind").selectOption("country:us");
    await wait("(()=>{const rows=[...document.querySelectorAll('#tbl tbody tr')].map(x=>x.textContent).join('');return rows.includes('미국')&&!rows.includes('북한')})()",9000);
    rows=(await page.locator("#tbl tbody tr").allInnerTexts()).join("\n");
    if(!rows.includes("미국")||rows.includes("북한"))throw new Error("US filter incorrect");
    await page.locator("#filter-kind").selectOption("");
    const downloadWait=page.waitForEvent("download",{timeout:15000});
    await page.locator("#export").click();
    const d=await downloadWait;
    const csv=await fs.readFile(await d.path(),"utf8");
    for(const name of ["미국","일본"])if(!csv.includes(name))throw new Error("CSV missing WORLD country "+name);
    if(!/(^|\r?\n)KR,/.test(csv)||!/(^|\r?\n)KP,/.test(csv))throw new Error("Existing KR/KP CSV codes changed");
    return {country_options:options,geocode_requests:requests.length,csv_bytes:Buffer.byteLength(csv),csv_kind_policy:"KR/KP codes unchanged; WORLD countries named"};
   });
   await check("file_replacement_removes_dynamic_kp_and_country",async ()=>{
    await page.locator("#filter-kind").selectOption("kp");
    await page.locator("#file").setInputFiles({name:"only-kr.txt",mimeType:"text/plain",buffer:Buffer.from("대한민국 위경도: 37.56, 126.98\n")});
    await page.locator("#parse").click();
    await wait("document.getElementById('coords-count').textContent.trim()==='1'",18000);
    const options=await page.locator("#filter-kind option").evaluateAll(list=>list.map(o=>o.value));
    if(JSON.stringify(options)!==JSON.stringify(["","kr","에러"]))throw new Error("Stale country option "+JSON.stringify(options));
    if(await page.locator("#filter-kind").inputValue()!=="")throw new Error("Stale filter selection");
    return {options};
   });
   await check("invalid_manual_input_reports_error",async ()=>{
    await page.locator("#manual-tab").click();
    await page.locator("#num64").fill("not-an-integer");
    await page.locator("#runBtn").click();
    await wait("document.getElementById('num64-error').textContent.trim().length>0",6000);
    return {error:(await page.locator("#num64-error").innerText()).slice(0,180)};
   });
   await check("no_unexpected_console_or_navigation_errors",async ()=>{
    const cancelledIndexPrefetches=requestFailures.filter(x=>
     x.failure?.errorText==="net::ERR_ABORTED" &&
     (x.url.startsWith("http://127.0.0.1:18973/data/korea-boundary-v4/index.bin?sha256=") ||
      x.url.startsWith("http://127.0.0.1:18973/data/korea-boundary-v4/tile-select-v5.bin?sha256=")));
    const unexpectedRequests=requestFailures.filter(x=>!cancelledIndexPrefetches.includes(x));
    if(pageErrors.length||consoleErrors.length||unexpectedRequests.length||cancelledIndexPrefetches.length>2)
     throw new Error(JSON.stringify({pageErrors,consoleErrors,requestFailures}).slice(0,1600));
    return {pageErrors:0,consoleErrors:0,unexpectedRequests:0,cancelledIndexPrefetches:cancelledIndexPrefetches.length,browserVersion:browser.version()};
   });
  }catch(e){console.error("ENGINE FAILURE "+name+" "+String(e))}
  finally{if(context)await context.close();if(browser)await browser.close()}
 }
}finally{
 if(server)await new Promise(resolve=>server.close(resolve));
 const totals={passed:cases.filter(x=>x.status==="passed").length,failed:cases.filter(x=>x.status==="failed").length,engines:[...new Set(cases.map(x=>x.engine))],failures};
 const outputPath=path.join(output,"three-browser-results.json");
 await fs.writeFile(outputPath,JSON.stringify({revision:"beb3a69d1cdcf7e23c8e64ce10f70e446027f20c",sourceSha256:expected,cases,summary:totals},null,2));
 console.log(JSON.stringify({path:outputPath,summary:totals}));
 if(totals.passed!==21||totals.failed)process.exitCode=1;
}