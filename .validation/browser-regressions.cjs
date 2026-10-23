const http = require('node:http');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const { chromium, firefox, webkit } = require('playwright');
const html = fs.readFileSync('revgeo.html');
const fixture = 'SRG\r\n대한민국 위경도: 37.5665,126.978\r\n세계 위경도: 95,200\r\n';
const successBody = JSON.stringify({
  type: 'FeatureCollection',
  features: [{
    properties: {
      geocoding: {
        label: '서울특별시 중구 세종대로 110',
        country_code: 'kr',
        name: '서울시청',
        type: 'public',
        admin: { level4: '서울특별시', level6: '중구' }
      }
    }
  }]
});
const host = http.createServer((req,res) => {
  if (req.url === '/' || req.url === '/revgeo.html') {
    res.writeHead(200, {'content-type':'text/html; charset=utf-8'});
    return res.end(html);
  }
  if (req.url.startsWith('/icon.svg')) {
    res.writeHead(200, {'content-type':'image/svg+xml'});
    return res.end('<svg xmlns="http://www.w3.org/2000/svg"/>');
  }
  res.writeHead(404);
  res.end();
});
const pause = ms => new Promise(resolve => setTimeout(resolve,ms));
async function setFile(page, contents) {
  await page.locator('#file').setInputFiles({
    name:'random_data.txt', mimeType:'text/plain', buffer:Buffer.from(contents)
  });
  await page.locator('#parse').click();
  await page.waitForFunction(() => document.getElementById('coords-count').textContent.trim() === '1', {timeout:10000});
  await page.locator('#start').waitFor({state:'visible'});
  assert.equal(await page.locator('#start').isDisabled(),false);
}
async function waitDone(page,id) {
  await page.waitForFunction(({id}) => document.getElementById(id).textContent.trim() === '1'
    && document.getElementById('start').hidden === false
    && document.querySelector('.status-row').getAttribute('data-state') !== 'running', {id}, {timeout:16000});
}
async function runFor(browserType,label,url) {
  const browser = await browserType.launch({headless:true});
  try {
    const context = await browser.newContext({viewport:{width:1280,height:850}, acceptDownloads:true});
    const errors = [];
    const page = await context.newPage();
    page.on('pageerror',err=>errors.push(err.message));
    let requests = 0;
    await page.route('https://nominatim.openstreetmap.org/**',async route=>{
      requests++;
      await route.fulfill({status:200,contentType:'application/json',
        headers:{'access-control-allow-origin':'*'},body:successBody});
    });
    await page.goto(url,{waitUntil:'domcontentloaded'});
    await setFile(page,fixture);
    await page.locator('#start').click();
    await waitDone(page,'cnt-success');
    assert.equal(requests,1,'first lookup must use Nominatim request');
    assert.match(await page.locator('#status').innerText(),/조회 완료/);
    await page.locator('#start').click();
    await waitDone(page,'cnt-success');
    assert.equal(requests,1,'repeat lookup must use in-memory cache');
    const other = await context.newPage();
    other.on('pageerror',err=>errors.push(err.message));
    let secondRequests = 0;
    let badResponse = false;
    let slowResponse = false;
    await other.route('https://nominatim.openstreetmap.org/**',async route=>{
      secondRequests++;
      if(slowResponse){await pause(1800);try{await route.fulfill({status:200,contentType:'application/json',headers:{'access-control-allow-origin':'*'},body:successBody});}catch{}return;}
      if(badResponse){await route.fulfill({status:403,contentType:'text/plain',headers:{'access-control-allow-origin':'*'},body:'Forbidden'});return;}
      await route.fulfill({status:200,contentType:'application/json',headers:{'access-control-allow-origin':'*'},body:successBody});
    });
    await other.goto(url,{waitUntil:'domcontentloaded'});
    await setFile(other,fixture);
    await other.locator('#start').click();
    await waitDone(other,'cnt-success');
    assert.equal(secondRequests,0,'another tab must load persistent IndexedDB cache');
    await other.evaluate(()=>{
      const channel = new BroadcastChannel('srg-revgeo-cache:v4');
      channel.postMessage([0,'test-other-tab']);
      channel.postMessage([2,'test-other-tab']);
      channel.postMessage([1,'test-other-tab']);
      channel.close();
    });
    await pause(80);
    assert.equal(errors.length,0,'broadcast channel must not produce unhandled errors');
    badResponse=true;
    await setFile(other,'대한민국 위경도: 35.1,128.1\n');
    await other.locator('#start').click();
    await waitDone(other,'cnt-fail');
    assert.equal(secondRequests,1,'failure path must issue one request');
    badResponse=false;
    await other.locator('#start').click();
    await waitDone(other,'cnt-success');
    assert.equal(secondRequests,2,'failed responses must not be cached');
    slowResponse=true;
    await setFile(other,'대한민국 위경도: 34.11,126.02\n');
    await other.locator('#start').click();
    await other.waitForRequest(r=>r.url().includes('34.11'),{timeout:12000});
    await other.locator('#stop').click();
    await other.waitForFunction(()=>document.querySelector('#status').textContent.includes('중지됨'),null,{timeout:8000});
    const afterStop=await other.locator('#cnt-success').innerText();
    await pause(1900);
    assert.equal(await other.locator('#cnt-success').innerText(),afterStop,'late network completion after abort must not update result');
    await other.setViewportSize({width:390,height:844});
    await other.locator('#mobile-results-tab').click();
    assert.equal(await other.locator('body').getAttribute('data-mobiletab'),'results');
    await other.locator('#mobile-input-tab').click();
    assert.equal(await other.locator('body').getAttribute('data-mobiletab'),'input');
    assert.deepEqual(errors,[],'no browser runtime exception');
    console.log(JSON.stringify({browser:label,lookupRequests:requests+secondRequests,
      cacheHits:2,abortedLookup:true,mobileNavigation:true,consoleExceptions:0}));
    await context.close();
  } finally {
    await browser.close();
  }
}
(async()=>{
  await new Promise(resolve=>host.listen(0,'127.0.0.1',resolve));
  const url='http://127.0.0.1:'+host.address().port+'/revgeo.html';
  try {
    for(const [name,engine] of [['chromium',chromium],['firefox',firefox],['webkit',webkit]]){
      await runFor(engine,name,url);
    }
  } finally {
    await new Promise(resolve=>host.close(resolve));
  }
})().catch(err=>{console.error(err);process.exitCode=1;});