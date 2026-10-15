"""Native browser checks against HTTP-served isolated baseline/candidate copies."""
import csv
import functools
import hashlib
import http.server
import io
import json
import statistics
import sys
import threading
from pathlib import Path
from playwright.sync_api import sync_playwright

HOOK = r'''
    globalThis.__auditYield = async (yieldFn) => {
      await yieldFn();
      const hook = globalThis.__audit.onYield;
      globalThis.__audit.onYield = null;
      hook?.();
    };
    globalThis.__audit = {
      AppState, CONFIG, domElements, parseFileToRecords, handleExport,
      get exportTask() { return EXPORT_TASK; },
      onYield: null, pauseAt: null, stopAt: null,
      async seed(mode, count) {
        clearTable();
        REVERSE_PARSED_CACHE.clear();
        const rows = Array.from({length: count}, (_, i) => {
          const latStr = String(35 + i / 100000), lonStr = '127';
          return {latStr, lonStr, set: 'WORLD', cacheKey: latStr + ',' + lonStr, result: null};
        });
        const parsed = {addr: {sido:'test', sigungu:'city', eupmyeon:'district',
          tag:'=SUM(1,2)', fullAddr: 'A'.repeat(300)}, kind:'WORLD'};
        const savedAt = Date.now() - 86400000;
        AppState.setRecords(rows);
        AppState.droppedFile = new File(['fixture'], 'audit.txt');
        AppState.bulkAcknowledged = true;
        if (mode === 'memory') {
          for (const row of rows) putReverseMemoryCache(row.cacheKey, parsed, savedAt);
        } else if (mode === 'persistent') {
          const db = await openReverseCacheDb();
          if (!db) throw new Error('Native IndexedDB is required by the test');
          await runReverseCacheTransaction(db, 'readwrite', new AbortController().signal,
            store => {for (const row of rows) store.put({key:REVERSE_CACHE_KEY_PREFIX + row.cacheKey, parsed, savedAt});});
        }
        return savedAt;
      },
      installPause() {
        this.onYield = () => {
          this.pauseAt = AppState.processed;
          domElements.pauseBtn.click();
        };
      },
      installStopDuringWrite() {
        const write = writePersistentReverseCacheBatch;
        writePersistentReverseCacheBatch = async (...args) => {
          const pending = write(...args);
          this.stopAt = AppState.processed;
          domElements.stopBtn.click();
          return pending;
        };
      },
      memoryTimestamps() { return [...REVERSE_PARSED_CACHE.values()].map(row => row.savedAt); },
      async measure(count) {
        await this.seed('memory', count);
        const task = AppState.reset('running');
        const started = performance.now();
        await runProcessingLoop(task);
        return {ms:performance.now()-started, processed:AppState.processed,
          rows:domElements.tbody.rows.length, pages:domElements.pageNumbers.children.length};
      }
    };
'''

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    baseline, candidate, evidence = map(Path, sys.argv[1:4])
    evidence.mkdir(parents=True, exist_ok=True)
    served = evidence / 'served'
    served.mkdir(exist_ok=True)
    for label, root in [('baseline', baseline), ('candidate', candidate)]:
        text = (root / 'revgeo.html').read_text(encoding='utf-8')
        text = text.replace('      initializeManualRng();',
            '      globalThis.__auditManual = {generateRandomDataFromNum, formatOutput, parseU64};\n      initializeManualRng();')
        text = (HOOK + '\n  </script>').join(text.rsplit('  </script>', 1))
        (served / f'{label}.html').write_text(text, encoding='utf-8')
        scheduled = text.replace('let lastYield = performance.now();', 'let lastYield = -Infinity;')
        scheduled = scheduled.replace('await yieldToMain();', 'await globalThis.__auditYield(yieldToMain);')
        (served / f'{label}-scheduled.html').write_text(scheduled, encoding='utf-8')
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Handler, directory=str(served)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f'http://127.0.0.1:{server.server_port}'
    results = {'playwright':'1.63.0', 'method':'native browsers, HTTP localhost, intercepted endpoint fixtures; scheduled copies force only the first cache yield', 'engines':{}}
    candidate_failures = []
    fixture = json.dumps({'features':[{'properties':{'geocoding':{'label':'fixture address', 'country_code':'kr', 'state':'test', 'city':'city'}}}]})
    with sync_playwright() as playwright:
        for engine in ['chromium', 'firefox', 'webkit']:
            browser = getattr(playwright, engine).launch()
            engine_result = {'version':browser.version, 'variants':{}, 'runtime_ms':{}}
            results['engines'][engine] = engine_result
            def new_page(label, scheduled=False):
                context = browser.new_context(accept_downloads=True, viewport={'width':1440,'height':1080})
                page = context.new_page()
                errors = []
                page.on('pageerror', lambda err: errors.append(str(err)))
                def route_request(route):
                    if route.request.url.startswith(origin):
                        route.continue_()
                    elif route.request.url.startswith('https://nominatim.openstreetmap.org/reverse?'):
                        route.fulfill(status=200, content_type='application/json', body=fixture,
                                      headers={'Access-Control-Allow-Origin':'*'})
                    else:
                        route.abort()
                page.route('**/*', route_request)
                page.goto(f'{origin}/{label}{"-scheduled" if scheduled else ""}.html')
                page.wait_for_function('!!globalThis.__audit')
                return context, page, errors
            for label in ['baseline','candidate']:
                cases = []
                engine_result['variants'][label] = cases
                def record(name, passed, detail):
                    cases.append({'name':name,'passed':bool(passed),'detail':detail})
                    print(engine, label, name, passed, detail, flush=True)
                    if label == 'candidate' and not passed:
                        candidate_failures.append(f'{engine}/{name}: {detail}')
                for mode in ['memory','persistent']:
                    context,page,errors = new_page(label,True)
                    saved_at = page.evaluate('mode => __audit.seed(mode,200)', mode)
                    page.evaluate('__audit.installPause()')
                    page.locator('#start').click()
                    page.wait_for_function('__audit.pauseAt !== null && ["paused","idle"].includes(__audit.AppState.status)')
                    observed = page.evaluate('({status:__audit.AppState.status,processed:__audit.AppState.processed,pauseAt:__audit.pauseAt})')
                    paused = observed['status'] == 'paused' and observed['processed'] < 200
                    if paused:
                        page.wait_for_timeout(30)
                        paused = paused and page.evaluate('__audit.AppState.processed') == observed['processed']
                        page.locator('#resume').click()
                        page.wait_for_function('__audit.AppState.status === "idle"')
                    observed['final'] = page.evaluate('__audit.AppState.processed')
                    observed['timestamp_preserved'] = page.evaluate('stamp => __audit.memoryTimestamps().every(v => v === stamp)', saved_at)
                    record(f'{mode}-pause-resume', paused and observed['final']==200 and observed['timestamp_preserved'] and not errors, observed)
                    context.close()
                context,page,errors = new_page(label)
                page.evaluate('__audit.seed("none",8)')
                page.evaluate('() => {__audit.CONFIG.RATE_LIMIT_DELAY_MS=1; __audit.installStopDuringWrite();}')
                page.locator('#start').click()
                page.wait_for_function('__audit.stopAt !== null && __audit.AppState.status === "idle"')
                observed = page.evaluate('({stoppedAt:__audit.stopAt,processed:__audit.AppState.processed})')
                record('no-result-after-stop-during-cache-write', observed['stoppedAt']==observed['processed'] and not errors, observed)
                context.close()
                context,page,errors = new_page(label)
                config = page.evaluate('({limit:__audit.CONFIG.BULK_HARD_LIMIT,delay:__audit.CONFIG.RATE_LIMIT_DELAY_MS,file:__audit.CONFIG.INPUT_FILE_HARD_LIMIT})')
                text = '\uc138\uacc4 \uc704\uacbd\ub3c4: 0, 0\r\n\ub300\ud55c\ubbfc\uad6d \uc704\uacbd\ub3c4: 35, 127\r\uc138\uacc4 \uc704\uacbd\ub3c4: -90, -180\n\uc138\uacc4 \uc704\uacbd\ub3c4: 91, 0\n'
                page.locator('#file').set_input_files({'name':'coords.txt','mimeType':'text/plain','buffer':text.encode()})
                page.locator('#parse').click()
                page.wait_for_function('__audit.AppState.status === "idle" && __audit.AppState.records.length > 0')
                parsed = page.evaluate('__audit.AppState.records.map(r=>[r.latStr,r.lonStr])')
                limits = page.evaluate('''async () => {
                  const test = async content => {try {await __audit.parseFileToRecords(new File([content],'test.txt'), new AbortController().signal);return false;}catch {return true;}};
                  return [await test(new Uint8Array([255])), await test('x'.repeat(1000001)), await test('\uC138\uACC4 \uC704\uACBD\uB3C4: 1, 2\\n'.repeat(5001))];
                }''')
                record('input-boundaries-and-default-limits', parsed==[['0','0'],['35','127'],['-90','-180']] and all(limits) and config=={'limit':5000,'delay':1000,'file':67108864} and not errors, {'parsed':parsed,'rejected':limits,'config':config,'errors':errors})
                page.evaluate('__audit.seed("memory",5000)')
                page.locator('#start').click()
                page.wait_for_function('__audit.AppState.status === "idle" && __audit.AppState.processed === 5000')
                bounds = page.evaluate('({rows:__audit.domElements.tbody.rows.length,pages:__audit.domElements.pageNumbers.children.length})')
                hashes=[]
                for repeat in range(2):
                    with page.expect_download() as download:
                        page.locator('#export').click()
                    content = Path(download.value.path()).read_bytes()
                    hashes.append(hashlib.sha256(content).hexdigest())
                rows=list(csv.reader(io.StringIO(content.decode('utf-8-sig'))))
                export_ok = len(rows)==5001 and all(len(row)==15 for row in rows) and rows[1][6].startswith("'=") and hashes[0]==hashes[1] and content.startswith(b'\xef\xbb\xbf')
                record('5000-row-pagination-and-repeated-csv-export', export_ok and bounds['rows']==10 and bounds['pages']<=7 and not errors, {'rows':len(rows)-1,'dom':bounds,'sha256':hashes,'errors':errors})
                if label=='candidate':
                    page.screenshot(path=str(evidence/f'{engine}-candidate.png'),full_page=True)
                context.close()
                context,page,errors=new_page(label,True)
                page.evaluate('__audit.seed("memory",20)')
                page.locator('#start').click()
                page.wait_for_function('__audit.AppState.status === "idle" && __audit.AppState.processed === 20')
                downloads=[]
                page.on('download',lambda d:downloads.append(d.suggested_filename))
                page.evaluate('() => { __audit.onYield = () => __audit.AppState.setRecords([]); }')
                page.locator('#export').click()
                page.wait_for_function('__audit.exportTask === null')
                page.wait_for_timeout(30)
                record('export-cancelled-on-generation-change',not downloads and not errors,{'downloads':downloads,'errors':errors})
                context.close()
            samples={'baseline':[],'candidate':[]}
            contexts={label:new_page(label) for label in samples}
            for trial in range(42):
                for label in (['baseline','candidate'] if trial%2==0 else ['candidate','baseline']):
                    _,page,errors=contexts[label]
                    sample=page.evaluate('__audit.measure(5000)')
                    if sample['processed']!=5000 or sample['rows']!=10 or sample['pages']>7 or errors:
                        candidate_failures.append(f'{engine}/{label}/runtime-invariant: {sample} {errors}')
                    if trial>=2:samples[label].append(sample['ms'])
            engine_result['runtime_ms']={label:{'samples':v,'median':statistics.median(v)} for label,v in samples.items()}
            for context,_,_ in contexts.values():context.close()
            browser.close()
    server.shutdown()
    (evidence/'browser-results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))
    if candidate_failures:
        raise SystemExit('\n'.join(candidate_failures))

if __name__=='__main__':
    main()
