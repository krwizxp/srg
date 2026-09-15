import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {createServer} from 'node:http';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
import {cpus,platform,release} from 'node:os';
import assert from 'node:assert/strict';
const work=process.env.GOAL_WORK,name=process.env.GOAL_ENGINE;
const engines=await import(process.env.GOAL_PLAYWRIGHT),phases=['baseline','candidate'],html={},hashes={};
for(const p of phases){html[p]=await readFile(resolve(work,p+'.html'),'utf8');hashes[p]=createHash('sha256').update(html[p]).digest('hex');}
const hook=`globalThis.__goal={CONFIG,REVERSE_PARSED_CACHE,readPersistentReverseCache,writePersistentReverseCacheBatch,syncChannel:REVERSE_CACHE_SYNC_CHANNEL,get externalWrites(){return REVERSE_CACHE_EXTERNAL_WRITES;}};`;
const server=createServer((req,res)=>{const phase=req.url.includes('candidate')?'candidate':'baseline';res.setHeader('content-type','text/html; charset=utf-8');res.end(html[phase].replace(/\n  <\/script>\n<\/body>/,`\n${hook}\n  </script>\n</body>`));});
await new Promise(r=>server.listen(0,'127.0.0.1',r));const origin=`http://127.0.0.1:${server.address().port}`;
const browser=await engines[name].launch({headless:true});
const plan={limit_pct:5,blocks:6,pairs_per_block:16,warmup_pairs_per_block:3,pilot_runs:3,target_batch_ms:500,max_multiplier:2048,base_repetitions:2,bootstrap_samples:10000,seed:20260915,statistic:'median paired percentage change; percentile bootstrap 95% CI',order:'alternate contexts per block, AB/BA per pair',scope:'actual BroadcastChannel plus persistent writes; keyed 100 reads; adaptive bulk 1500 reads; unchanged CSV algorithms retain earlier evidence',memory:'functional token payload bound 16384 bytes, no new timer; browser RSS in verify.mjs'};
const cases=['coordinated-write','keyed-read-100','adaptive-read-1500'],results={},pageErrors=[];
async function open(phase){const context=await browser.newContext();const page=await context.newPage();page.on('pageerror',e=>pageErrors.push(e.message));await page.route('https://fonts.googleapis.com/**',r=>r.fulfill({body:'',contentType:'text/css'}));await page.goto(origin+'/'+phase);await page.waitForFunction(()=>!!globalThis.__goal);
 await page.evaluate(async()=>{const g=__goal;globalThis.__parsed={addr:{sido:'s',sigungu:'g',eupmyeon:'e',tag:'t',fullAddr:'cache fixture'},kind:'KR'};globalThis.__signal=new AbortController().signal;
  globalThis.__entries=Array.from({length:1500},(_,i)=>({cacheKey:'fixture-'+i,parsed:__parsed}));await g.writePersistentReverseCacheBatch(__entries,__signal);g.REVERSE_PARSED_CACHE.clear();
  globalThis.__peer=new BroadcastChannel(`${g.CONFIG.REVERSE_CACHE_DB_NAME}:v${g.CONFIG.REVERSE_CACHE_DB_VERSION}`);
  globalThis.__send=data=>new Promise(resolve=>{const listener=({data:received})=>{if(received[0]===data[0]&&received[1]===data[1]){g.syncChannel.removeEventListener('message',listener);resolve();}};g.syncChannel.addEventListener('message',listener);__peer.postMessage(data);});
 });return {context,page};}
async function measure(page,label,multiplier){return page.evaluate(async({label,multiplier})=>{const g=__goal;let checksum=0;const start=performance.now();for(let i=0;i<2*multiplier;i++){
 if(label==='coordinated-write'){await __send([0,'normal-peer']);if(g.externalWrites?.size!==1)throw Error('missing peer');await g.writePersistentReverseCacheBatch(__entries.slice(0,32),__signal);await __send([1,'normal-peer']);if(g.externalWrites?.size!==0)throw Error('peer retained');checksum+=32;}
 else {const n=label==='keyed-read-100'?100:1500;g.REVERSE_PARSED_CACHE.clear();const map=new Map(__entries.slice(0,n).map(e=>[e.cacheKey,null]));await g.readPersistentReverseCache(map,__signal);for(const v of map.values()){if(v?.addr.fullAddr!=='cache fixture')throw Error('cache mismatch');checksum++;}}
 }return {ms:performance.now()-start,checksum};},{label,multiplier});}
function median(xs){const v=[...xs].sort((a,b)=>a-b),n=v.length;return (v[Math.floor((n-1)/2)]+v[Math.floor(n/2)])/2;}
function analyze(samples){const values=samples.map(p=>(p.candidate.ms/p.baseline.ms-1)*100);let seed=plan.seed;const rand=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};const boot=Array.from({length:plan.bootstrap_samples},()=>median(values.map(()=>values[Math.floor(rand()*values.length)]))).sort((a,b)=>a-b);return {pairs:values.length,baselineMedianMs:median(samples.map(p=>p.baseline.ms)),candidateMedianMs:median(samples.map(p=>p.candidate.ms)),medianChangePct:median(values),ci95:[boot[249],boot[9749]],pass:boot[9749]<=plan.limit_pct};}
try{for(const label of cases){const samples=[],pilots=[],multipliers=[];for(let block=0;block<plan.blocks;block++){const pages={},creation=block%2?phases.toReversed():phases;try{for(const p of creation)pages[p]=await open(p);
 const times=[];for(let i=0;i<plan.pilot_runs;i++)for(const p of i%2?creation.toReversed():creation){const v=await measure(pages[p].page,label,1);times.push(Math.max(1,v.ms));pilots.push({block,phase:p,...v});}
 const multiplier=Math.min(plan.max_multiplier,Math.max(1,Math.ceil(plan.target_batch_ms/median(times))));multipliers.push(multiplier);
 for(let i=-plan.warmup_pairs_per_block;i<plan.pairs_per_block;i++){const pair={block,index:i,multiplier};for(const p of (i+block)%2?phases.toReversed():phases)pair[p]=await measure(pages[p].page,label,multiplier);assert.equal(pair.baseline.checksum,pair.candidate.checksum);if(i>=0)samples.push(pair);}
 }finally{for(const p of Object.values(pages))await p.context.close();}console.log('GOAL_PERF_BLOCK '+JSON.stringify({label,block,pairs:samples.length}));}
 results[label]={pilots,multipliers,samples,analysis:analyze(samples)};
 }
 assert.deepEqual(pageErrors,[]);const report={engine:name,version:browser.version(),os:platform()+' '+release(),cpu:cpus()[0].model,node:process.version,hashes,plan,results,pageErrors};await mkdir(resolve(work,'results'),{recursive:true});await writeFile(resolve(work,'results',name+'-cache-perf.json'),JSON.stringify(report,null,2));console.log('GOAL_CACHE_JSON='+JSON.stringify(report));if(!Object.values(results).every(v=>v.analysis.pass))process.exitCode=1;
}finally{await browser.close();server.close();}
