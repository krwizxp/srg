"""Temporary optical and appearance checks against pinned production pages."""
import functools
import http.server
import importlib.metadata
import json
import statistics
import sys
import threading
from pathlib import Path
from PIL import Image
from playwright.sync_api import sync_playwright

BASELINE_OPAQUE_MAX_DIFFERENCE = 2
GLASS_MIN_DIFFERENCE = 8
EXPECTED = {'light': 'rgb(245, 245, 247)', 'dark': 'rgb(17, 17, 19)'}

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def end_headers(self):
        self.send_header('Content-Security-Policy', "style-src-elem 'self' 'unsafe-inline'")
        super().end_headers()

def channel_medians(path, box):
    with Image.open(path).convert('RGB') as image:
        region = image.crop(tuple(round(v) for v in box))
        pixels = list(region.getdata())
        return [statistics.median(pixel[channel] for pixel in pixels) for channel in range(3)]

def main():
    baseline, candidate, evidence = [Path(v).resolve() for v in sys.argv[1:4]]
    evidence.mkdir(parents=True, exist_ok=True)
    assert baseline.parent == candidate.parent
    handler = functools.partial(Handler, directory=str(baseline.parent))
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f'http://127.0.0.1:{server.server_port}'
    data = {
        'playwright': importlib.metadata.version('playwright'),
        'pillow': importlib.metadata.version('Pillow'),
        'fixture': 'Pristine source; optional external font blocked; test underlay only for optical pairs.',
        'thresholds': {'opaque_max_channel_difference': BASELINE_OPAQUE_MAX_DIFFERENCE,
                       'glass_min_channel_difference': GLASS_MIN_DIFFERENCE},
        'states': [], 'optics': [], 'engines': {}
    }
    try:
        with sync_playwright() as pw:
            for engine in ['chromium', 'firefox', 'webkit']:
                browser = getattr(pw, engine).launch()
                data['engines'][engine] = browser.version
                for variant in ['baseline', 'candidate']:
                    for system in ['light', 'dark']:
                        context = browser.new_context(viewport={'width':393,'height':852},
                            color_scheme=system, device_scale_factor=1)
                        page = context.new_page()
                        errors = []
                        page.on('pageerror', lambda error: errors.append(str(error)))
                        page.route('**/*', lambda route: route.continue_() if route.request.url.startswith(origin) else route.abort())
                        page.goto(f'{origin}/{variant}/revgeo.html')
                        page.wait_for_function('!!document.querySelector("#theme-toggle")')
                        def snapshot(phase, expected, safe, mobile=True):
                            state = page.evaluate("""() => {
                                const wrap = document.querySelector('.mobile-bottom');
                                const nav = document.querySelector('.mobile-nav');
                                const control = nav.querySelector('[data-mobile="results"]');
                                return {
                                    page: getComputedStyle(document.body).backgroundColor,
                                    wrapper: getComputedStyle(wrap).backgroundColor,
                                    display: getComputedStyle(wrap).display,
                                    blur: getComputedStyle(nav).backdropFilter,
                                    button: {width:control.getBoundingClientRect().width,
                                             height:control.getBoundingClientRect().height},
                                    dom:document.querySelectorAll('*').length
                                };
                            }""")
                            assert state['page'] == EXPECTED[expected], state
                            if mobile:
                                assert state['display'] != 'none', state
                                assert state['button']['width'] >= 44 and state['button']['height'] >= 44, state
                                assert 'blur(20px)' in state['blur'], state
                            else:
                                assert state['display'] == 'none', state
                            assert not errors, errors
                            data['states'].append(dict(engine=engine,variant=variant,system=system,
                                phase=phase,safe=safe,expected=expected,**state))
                        for safe in [0,34]:
                            page.add_style_tag(content=f'.mobile-bottom {{padding-bottom:max(6px,{safe}px);}}')
                            snapshot('initial',system,safe)
                            page.locator('#theme-toggle').click()
                            snapshot('opposite','dark' if system=='light' else 'light',safe)
                            page.locator('#theme-toggle').click()
                            snapshot('returned',system,safe)
                        # The marker is temporary instrumentation, behind the actual unchanged capsule.
                        page.evaluate("""() => {
                            const marker = document.createElement('div');
                            marker.id = 'optical-underlay';
                            marker.style.cssText = 'position:fixed;left:0;right:0;bottom:12px;height:90px;z-index:24;pointer-events:none';
                            document.body.append(marker);
                        }""")
                        button = page.locator('.mobile-nav [data-mobile="results"]')
                        assert button.get_attribute('aria-pressed') == 'false'
                        rect = button.bounding_box()
                        box = (rect['x']+.82*rect['width'],rect['y']+.35*rect['height'],
                               rect['x']+.88*rect['width'],rect['y']+.65*rect['height'])
                        colors = {}
                        for shade,color in [('red','#e00000'),('blue','#0040f0')]:
                            page.locator('#optical-underlay').evaluate('(el,color) => el.style.background = color',color)
                            page.wait_for_timeout(200)
                            target = evidence/f'{engine}-{variant}-{system}-{shade}.png'
                            page.screenshot(path=str(target))
                            colors[shade] = channel_medians(target,box)
                        differences = [abs(a-b) for a,b in zip(colors['red'],colors['blue'])]
                        optical = dict(engine=engine,variant=variant,system=system,roi=box,
                            medians=colors,differences=differences,threshold=GLASS_MIN_DIFFERENCE if variant=='candidate' else BASELINE_OPAQUE_MAX_DIFFERENCE)
                        data['optics'].append(optical)
                        if variant == 'candidate':
                            assert max(differences) >= GLASS_MIN_DIFFERENCE, optical
                        else:
                            assert max(differences) <= BASELINE_OPAQUE_MAX_DIFFERENCE, optical
                        page.locator('#optical-underlay').evaluate('(el) => el.remove()')
                        page.evaluate('window.scrollTo(0,document.body.scrollHeight)')
                        assert page.locator('.mobile-nav').is_visible()
                        page.set_viewport_size({'width':852,'height':393})
                        snapshot('landscape',system,34)
                        page.set_viewport_size({'width':1440,'height':900})
                        snapshot('desktop',system,34,False)
                        context.close()
                browser.close()
    finally:
        server.shutdown()
        (evidence/'web-results.json').write_text(json.dumps(data,indent=2))
    print(json.dumps({'states':len(data['states']),'optical_pairs':len(data['optics']),
                      'engines':data['engines'],'optics':data['optics']},indent=2))

if __name__ == '__main__':
    main()
