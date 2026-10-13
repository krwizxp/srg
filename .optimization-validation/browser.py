"""Run unchanged revgeo.html in actual stable browsers via W3C WebDriver."""
import argparse, functools, http.server, json, os, pathlib, shutil, subprocess, threading, time, urllib.request

P=pathlib.Path; root=P.cwd(); out=root/'browser-results';out.mkdir(exist_ok=True)
result={'checks':[],'page_sha256':__import__('hashlib').sha256((root/'revgeo.html').read_bytes()).hexdigest()}
class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*args): pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(root)))
threading.Thread(target=server.serve_forever,daemon=True).start()
parser=argparse.ArgumentParser();parser.add_argument('--browser',required=True);kind=parser.parse_args().browser
result['browser_requested']=kind; driver=None; session=None
def request(method,path,payload=None):
    data=None if payload is None else json.dumps(payload).encode()
    req=urllib.request.Request('http://127.0.0.1:4444'+path,data=data,method=method,headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=70) as r: value=json.load(r)['value']
    except urllib.error.HTTPError as e: raise RuntimeError(e.read().decode()) from e
    if isinstance(value,dict) and value.get('error'):raise RuntimeError(str(value))
    return value
def cmd(path,data):return request('POST','/session/'+session+path,data)
def js(script,*args):return cmd('/execute/sync',{'script':script,'args':list(args)})
def wait(script,timeout=15):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        value=js('return '+script)
        if value:return value
        time.sleep(.1)
    raise AssertionError('Timed out: '+script+'; '+str(js('return document.querySelector("#status")?.textContent')))
def check(name,condition):
    assert condition,name;result['checks'].append(name)
def click(selector):
    e=cmd('/element',{'using':'css selector','value':selector});cmd('/element/'+e['element-6066-11e4-a52e-4f735466cecf']+'/click',{})
def upload(text):
    js('const d=new DataTransfer();d.items.add(new File([arguments[0]],"fixture.txt",{type:"text/plain"}));const e=document.querySelector("#file");e.files=d.files;e.dispatchEvent(new Event("change",{bubbles:true}));',text)
    click('#parse');wait('document.querySelector("#start").disabled===false')
def install_probe():
    js('''window.__errors=[];window.__fetches=0;window.__delay=0;window.__blobs=[];window.__revoked=[];
    addEventListener('error',e=>window.__errors.push(e.message));addEventListener('unhandledrejection',e=>window.__errors.push(String(e.reason)));
    window.fetch=async(url,options={})=>{window.__fetches++;await new Promise((resolve,reject)=>{const timer=setTimeout(resolve,window.__delay);options.signal?.addEventListener('abort',()=>{clearTimeout(timer);reject(options.signal.reason)},{once:true});});options.signal?.throwIfAborted();
    return new Response(JSON.stringify({features:[{properties:{geocoding:{label:'서울특별시, 종로구, 대한민국',country_code:'kr',city:'서울특별시',district:'종로구',name:'테스트 위치',type:'city'}}}]}),{headers:{'content-type':'application/json'}});};
    const create=URL.createObjectURL.bind(URL),revoke=URL.revokeObjectURL.bind(URL);
    URL.createObjectURL=blob=>{const url=create(blob);blob.text().then(text=>window.__blobs.push({url,text}));return url;};
    URL.revokeObjectURL=url=>{window.__revoked.push(url);revoke(url);};
    document.addEventListener('click',e=>{if(e.target.closest('a[download]'))e.preventDefault();},true);''')
try:
    if kind=='chrome':
        binary=str(root/'chrome-linux64/chrome'); executable=str(root/'chromedriver-linux64/chromedriver')
        options={'browserName':'chrome','goog:chromeOptions':{'binary':binary,'args':['--headless=new','--no-sandbox','--no-first-run','--disable-background-networking','--disable-component-update','--disable-sync','--window-size=1440,1200']}}
        launch=[executable,'--port=4444','--verbose']
    elif kind=='firefox':
        executable=shutil.which('geckodriver') or str(P(os.environ['GECKOWEBDRIVER'])/'geckodriver')
        prefs={'datareporting.healthreport.uploadEnabled':False,'datareporting.policy.dataSubmissionEnabled':False,'toolkit.telemetry.enabled':False,'toolkit.telemetry.unified':False,'geo.enabled':False,'geo.provider.network.url':'','network.proxy.type':1,'network.proxy.http':'127.0.0.1','network.proxy.http_port':9,'network.proxy.ssl':'127.0.0.1','network.proxy.ssl_port':9,'network.proxy.no_proxies_on':'localhost, 127.0.0.1'}
        options={'browserName':'firefox','moz:firefoxOptions':{'binary':str(root/'firefox/firefox'),'args':['-headless'],'prefs':prefs}}
        launch=[executable,'--port','4444']
    else:
        options={'browserName':'safari'};launch=['/usr/bin/safaridriver','-p','4444']
    log=(out/'driver.log').open('w');driver=subprocess.Popen(launch,stdout=log,stderr=subprocess.STDOUT)
    for _ in range(100):
        try:request('GET','/status');break
        except Exception:time.sleep(.1)
    value=request('POST','/session',{'capabilities':{'alwaysMatch':options}});session=value['sessionId'];result['capabilities']=value['capabilities']
    version=value['capabilities']['browserVersion']
    check('stable-browser-version',version=={'chrome':'155.0.8059.39','firefox':'157.0.1'}.get(kind, version))
    cmd('/timeouts',{'script':60000,'pageLoad':60000,'implicit':0});cmd('/window/rect',{'width':1440,'height':1200})
    url='http://127.0.0.1:'+str(server.server_port)+'/revgeo.html';cmd('/url',{'url':url})
    wait('document.querySelector("#parse")!==null');install_probe()
    result['apis']=js('return {promiseTry:typeof Promise.try,promiseResolvers:typeof Promise.withResolvers,mapInsert:typeof Map.prototype.getOrInsert,toHex:typeof Uint8Array.prototype.toHex,iteratorMap:typeof Iterator.prototype.map,invoker:"commandForElement" in HTMLButtonElement.prototype,popover:typeof HTMLElement.prototype.showPopover,decoderStream:typeof TextDecoderStream,abortAny:typeof AbortSignal.any}')
    check('required-native-apis',all(v in [True,'function'] for v in result['apis'].values()))
    before=js('return document.querySelector("#theme-toggle").getAttribute("aria-pressed")');click('#theme-toggle')
    check('theme-accessible-state',js('return document.querySelector("#theme-toggle").getAttribute("aria-pressed")')!=before)
    click('[popovertarget="display-settings"]');check('settings-popover',js('return document.querySelector("#display-settings").matches(":popover-open")'));click('[popovertarget="display-settings"]')
    fixture='대한민국 위경도: 37.5, 127.0\r\n세계 위경도: -20.5, 150.2\r세계 위경도: 100, 10\n'
    upload(fixture);check('coordinate-parse-boundary',js('return document.querySelector("#coords .coords-content").textContent')=='37.5, 127.0\n-20.5, 150.2')
    click('#start');wait('document.querySelector("#cnt-success").textContent==="2"');wait('document.querySelector("#start").disabled===false')
    wait('document.querySelectorAll("#tbl tbody tr").length===2');check('native-command-fetch-render',js('return window.__fetches===2'))
    js('const e=document.querySelector("#filter-q");e.value="no-such-fixture";e.dispatchEvent(new Event("input",{bubbles:true}));')
    wait('document.querySelectorAll("#tbl tbody tr").length===0');click('#filter-clear');wait('document.querySelectorAll("#tbl tbody tr").length===2');check('filter-clear',True)
    click('#export');wait('window.__blobs.length===1');check('csv-content',js('return window.__blobs[0].text.includes("서울특별시") && window.__blobs[0].text.includes("37.5")'))
    click('#export');wait('window.__blobs.length===2');check('download-url-cleanup',js('return window.__revoked.includes(window.__blobs[0].url)'))
    # A fresh document must read the origin's persistent cache, with no mock network calls.
    cmd('/url',{'url':url});wait('document.querySelector("#parse")!==null');install_probe();upload(fixture);click('#start');wait('document.querySelector("#cnt-success").textContent==="2"');wait('document.querySelector("#start").disabled===false');check('persistent-cache-reload',js('return window.__fetches===0'))
    upload('세계 위경도: 1, 2\n세계 위경도: 3, 4\n세계 위경도: 5, 6');js('window.__delay=500');click('#start');click('#pause');wait('document.querySelector("#resume").disabled===false');check('pause',True);click('#resume');wait('document.querySelector("#stop").disabled===false');click('#stop');wait('document.querySelector("#start").disabled===false');check('resume-stop-cancellation',True)
    completed=js('return document.querySelector("#cnt-success").textContent');time.sleep(1);check('no-stale-completion',js('return document.querySelector("#cnt-success").textContent')==completed)
    js('const e=document.querySelector("details.manual-rng");e.open=true;');wait('document.querySelector("#runBtn").disabled===false')
    js('document.querySelector("#num64").value="1234567890123456789";document.querySelector("#suppValues").value=Array(128).fill("9876543210987654321").join(" ");');click('#runBtn');wait('document.querySelector("#saveOutput").disabled===false');check('manual-rng-modern-byte-api',js('return document.querySelector("#output").textContent.includes("위경도")'));click('#clearBtn');check('manual-clear',js('return document.querySelector("#output .coords-content").textContent==="" && document.querySelector("#saveOutput").disabled'))
    result['errors']=js('return window.__errors');check('no-page-errors',not result['errors']);result['status']='passed'
except BaseException as e:
    result.update(status='failed',error=str(e))
    if session:
        try:result['failure_state']=js('return {status:document.querySelector("#status")?.textContent,fetches:window.__fetches,rows:document.querySelectorAll("#tbl tbody tr").length,errors:window.__errors}')
        except Exception:pass
    raise
finally:
    if session:
        try:request('DELETE','/session/'+session)
        except Exception:pass
    if driver:driver.terminate();driver.wait(timeout=10)
    server.shutdown();(out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
