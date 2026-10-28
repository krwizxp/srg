const fs=require('fs'),assert=require('assert/strict');
const {chromium,firefox,webkit}=require('playwright');
const engine=process.env.SRG_ENGINE||'chromium';
const bytes=fs.readFileSync('tools/validation/values.bin');
const values=Array.from({length:1000},(_,i)=>bytes.readBigUInt64LE(i*8).toString());
const supplements=Array.from({length:3000},(_,i)=>bytes.readBigUInt64LE((i+1000)*8).toString());
async function prepare(page,url,nums=values){
 await page.goto(url);await page.locator('#manual-tab').click();
 await page.evaluate(({nums,supplements})=>{
  for(const [id,value]of[['num64',nums.join('\n')],['suppValues',supplements.join('\n')]]){const e=document.getElementById(id);e.value=value;e.dispatchEvent(new Event('input',{bubbles:true}));}
 },{nums,supplements});
}
async function convert(page){await page.locator('#runBtn').click();await page.waitForFunction(()=>document.getElementById('meta').textContent.includes('완료')&&!document.getElementById('runBtn').disabled);assert.equal(await page.locator('#error').textContent(),'');}
(async()=>{
 const browser=await({chromium,firefox,webkit}[engine]).launch({headless:true});const context=await browser.newContext({acceptDownloads:true});
 try{
  const outputs=[];const errors=[];
  for(const prefix of ['/baseline-validation','']){
   const page=await context.newPage();page.setDefaultTimeout(60000);page.on('pageerror',e=>errors.push(e.message));
   await prepare(page,'http://127.0.0.1:8765'+prefix+'/revgeo.html');await convert(page);
   const wait=page.waitForEvent('download');await page.locator('#saveOutput').click();const download=await wait;outputs.push(fs.readFileSync(await download.path()));
  }
  assert.deepEqual(outputs[0],outputs[1]);
  const invalid=await context.newPage();let boundaryRequests=0;invalid.on('request',r=>{if(r.url().includes('korea_boundary')||r.url().includes('/data/'))boundaryRequests++});
  await prepare(invalid,'http://127.0.0.1:8765/revgeo.html',['18446744073709551616']);await invalid.locator('#runBtn').click();await invalid.waitForFunction(()=>document.getElementById('num64-error').textContent.length>0);assert.equal(boundaryRequests,0);
  const broken=await context.newPage();await broken.route('**/index.bin',route=>route.fulfill({status:503,body:'Unavailable'}));
  await prepare(broken,'http://127.0.0.1:8765/revgeo.html',['0']);await broken.locator('#runBtn').click();await broken.waitForFunction(()=>document.getElementById('error').textContent.includes('503'));assert.equal(await broken.locator('#runBtn').isEnabled(),true);assert.equal(await broken.locator('#output').textContent(),'');
  await broken.unroute('**/index.bin');await convert(broken);
  const standalone=await context.newPage();await standalone.route('**/korea_boundary.js',route=>route.abort());await standalone.goto('http://127.0.0.1:8765/revgeo.html');await standalone.locator('#file').setInputFiles({name:'random_data.txt',mimeType:'text/plain',buffer:Buffer.from('대한민국 위경도: 37.5,127.0\n세계 위경도: 35.0,135.0\n')});await standalone.getByRole('button',{name:'좌표 추출',exact:true}).click();await standalone.waitForFunction(()=>document.getElementById('coords-count').textContent==='2');
  const stale=await context.newPage();let release,seen;const hold=new Promise(r=>release=r),requested=new Promise(r=>seen=r);
  await stale.route('**/index.bin',async route=>{seen();await hold;await route.fulfill({status:200,body:fs.readFileSync('data/korea-boundary-v1/index.bin')})});
  await prepare(stale,'http://127.0.0.1:8765/revgeo.html',['0']);await stale.locator('#runBtn').click();await requested;await stale.locator('#clearBtn').click();release();await stale.waitForTimeout(500);assert.equal(await stale.locator('#output').textContent(),'');assert.equal(await stale.locator('#meta').textContent(),'');assert.equal(await stale.locator('#runBtn').isEnabled(),true);
  assert.deepEqual(errors,[]);console.log(JSON.stringify({engine,version:browser.version(),full_output_cases:1000,exact_output_bytes:true,invalidInputLazyLoad:true,missingDataAndRetry:true,standaloneAnalysis:true,staleCompletion:true}));
 }finally{await browser.close();}
})().catch(e=>{console.error(String(e).split('\n')[0]);process.exitCode=1});
