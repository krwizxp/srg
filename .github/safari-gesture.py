"""Temporary actual-touch Safari scroll reproduction; no product instrumentation."""
import functools
import hashlib
import http.server
import json
import subprocess
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
WORK = REPO / 'safari27-work'
EVIDENCE = REPO / 'safari27-evidence'
BASELINE = 'f36877f0c843edd6a71d19a3ff8ea3cbbb996ee7'
VARIANTS = ['baseline', 'inset8', 'inset24']

PROJECT = 'name: SafariProbe\noptions:\n  bundleIdPrefix: dev.srg.validation\n  deploymentTarget:\n    iOS: "27.0"\nsettings:\n  base:\n    SWIFT_VERSION: "5.0"\n    CODE_SIGNING_ALLOWED: NO\n    GENERATE_INFOPLIST_FILE: YES\ntargets:\n  SafariProbe:\n    type: application\n    platform: iOS\n    sources: [Launcher]\n    settings:\n      base:\n        TARGETED_DEVICE_FAMILY: "1"\n        INFOPLIST_KEY_UILaunchScreen_Generation: YES\n  SafariProbeUITests:\n    type: bundle.ui-testing\n    platform: iOS\n    sources: [UITests]\n    dependencies:\n      - target: SafariProbe\nschemes:\n  SafariProbe:\n    build:\n      targets:\n        SafariProbe: all\n    test:\n      targets: [SafariProbeUITests]\n'

LAUNCHER = 'import SwiftUI\nimport UIKit\n\n@main\nstruct SafariProbeApp: App {\n    var body: some Scene {\n        WindowGroup {\n            Text("Native Safari verification launcher")\n                .onAppear {\n                    let args = ProcessInfo.processInfo.arguments\n                    guard let index = args.firstIndex(of: "--url"),\n                          args.indices.contains(index + 1),\n                          let url = URL(string: args[index + 1]) else { return }\n                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {\n                        UIApplication.shared.open(url)\n                    }\n                }\n        }\n    }\n}\n'
UI_TEST = r'''import XCTest

final class SafariGestureCapture: XCTestCase {
    override func setUpWithError() throws { continueAfterFailure = false }

    func capture(_ phase: String, _ safari: XCUIApplication) {
        let shot = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
        shot.name = "\(ProbeConfig.name)__\(phase)"
        shot.lifetime = .keepAlways
        add(shot)
        let tree = XCTAttachment(string: safari.debugDescription)
        tree.name = "\(ProbeConfig.name)__\(phase)__accessibility"
        tree.lifetime = .keepAlways
        add(tree)
    }

    func open(_ url: String, _ safari: XCUIApplication) {
        let launcher = XCUIApplication()
        launcher.terminate()
        launcher.launchArguments = ["--url", url]
        launcher.launch()
        XCTAssertTrue(safari.wait(for: .runningForeground, timeout: 20))
    }

    func drag(_ up: Bool, _ safari: XCUIApplication) {
        let start = safari.coordinate(withNormalizedOffset: CGVector(dx: 0.52, dy: up ? 0.64 : 0.46))
        let end = safari.coordinate(withNormalizedOffset: CGVector(dx: 0.52, dy: up ? 0.46 : 0.64))
        start.press(forDuration: 0.1, thenDragTo: end,
                    withVelocity: XCUIGestureVelocity.slow, thenHoldForDuration: 0.15)
        Thread.sleep(forTimeInterval: 1)
    }

    func tap(_ label: String, _ safari: XCUIApplication) {
        let matches = safari.buttons.matching(NSPredicate(format: "label == %@", label)).allElementsBoundByIndex
        guard let button = matches.first(where: { $0.isHittable }) else {
            XCTFail("No visible normal-flow fixture button: \(label)")
            return
        }
        button.tap()
        Thread.sleep(forTimeInterval: 1)
    }

    func testRealTouchScrollAndAppearance() throws {
        let safari = XCUIApplication(bundleIdentifier: "com.apple.mobilesafari")
        safari.terminate()
        open(ProbeConfig.url, safari)
        for _ in 0..<8 {
            let toggle = safari.descendants(matching: .any).matching(NSPredicate(format: "label == %@", "다크 모드")).firstMatch
            if toggle.exists { break }
            for title in ["Continue", "Start Browsing", "Not Now", "Continue Without Extensions", "Open"] {
                let button = safari.buttons[title].firstMatch
                if button.exists && button.isHittable { button.tap() }
            }
            Thread.sleep(forTimeInterval: 0.5)
        }
        let toggle = safari.descendants(matching: .any).matching(NSPredicate(format: "label == %@", "다크 모드")).firstMatch
        XCTAssertTrue(toggle.waitForExistence(timeout: 30))
        Thread.sleep(forTimeInterval: 2)
        capture("pristine-initial", safari)
        drag(true, safari)
        capture("pristine-up", safari)
        drag(false, safari)
        for _ in 0..<4 {
            if toggle.isHittable { break }
            drag(false, safari)
        }
        capture("pristine-down", safari)
        XCTAssertTrue(toggle.isHittable)
        toggle.tap()
        Thread.sleep(forTimeInterval: 2)
        capture("pristine-opposite", safari)
        toggle.tap()
        Thread.sleep(forTimeInterval: 2)
        capture("pristine-returned", safari)

        open(ProbeConfig.url.replacingOccurrences(of: "revgeo.html", with: "gesture.html"), safari)
        let ready = safari.staticTexts["Gesture fixture ready"].firstMatch
        XCTAssertTrue(ready.waitForExistence(timeout: 30))
        Thread.sleep(forTimeInterval: 2)
        capture("fixture-initial", safari)
        let landmark = safari.staticTexts["Probe row 2"].firstMatch
        let before = landmark.frame.minY
        drag(true, safari)
        let after = landmark.frame.minY
        let movement = XCTAttachment(string: "landmarkBefore=\(before) landmarkAfter=\(after) displacement=\(after-before)")
        movement.name = "\(ProbeConfig.name)__gesture-movement"
        movement.lifetime = .keepAlways
        add(movement)
        XCTAssertGreaterThan(abs(after-before), 30, "Actual native touch must move the document")
        capture("fixture-up", safari)
        drag(false, safari)
        capture("fixture-down", safari)
        for theme in ["dark", "light"] {
            tap("Probe plain", safari)
            tap("Probe \(theme)", safari)
            capture("after-scroll-\(theme)", safari)
            tap("Probe pattern a", safari)
            capture("after-scroll-\(theme)-a", safari)
            drag(true, safari)
            drag(false, safari)
            tap("Probe pattern b", safari)
            capture("after-scroll-\(theme)-b", safari)
        }
    }
}
'''


def run(*args):
    return subprocess.check_output(args, cwd=REPO)


def apply_variant(source, variant):
    if variant == 'baseline':
        return source
    anchor = 'position: fixed; z-index: 25; bottom: 0; left: 0; right: 0; display: flex;'
    assert source.count(anchor) == 1
    offset = {'inset8': 8, 'inset24': 24}[variant]
    return source.replace(anchor, f'position: fixed; z-index: 25; bottom: {offset}px; left: 0; right: 0; display: flex;')


def prepare():
    WORK.mkdir(exist_ok=True)
    EVIDENCE.mkdir(exist_ok=True)
    source = run('git', 'show', f'{BASELINE}:revgeo.html').decode()
    assert hashlib.sha256(source.encode()).hexdigest() == '437e402c28e8f55d77af15dc94b87b096cb17c4323b447a5b8524e8a2ece725d'
    identities = {'baseline_commit': BASELINE, 'variants': {}, 'thresholds': {
        'native_pattern_max_channel_difference_min': 24,
        'native_pattern_left_right_contrast_min': 16,
        'native_pristine_theme_max_channel_error': 40,
        'native_edge_roi_x': [[.005,.02],[.98,.995]],
        'native_edge_roi_y': [.92,.965],
        'actual_native_gesture_landmark_movement_min_points': 30}}
    for variant in VARIANTS:
        destination = WORK / 'www' / variant
        destination.mkdir(parents=True, exist_ok=True)
        for path in ['icon.svg', 'site.webmanifest']:
            (destination/path).write_bytes(run('git', 'show', f'{BASELINE}:{path}'))
        html = apply_variant(source, variant)
        (destination/'revgeo.html').write_text(html)
        rows = ''.join(f'''<div style="height:420px;padding:20px;max-width:340px;margin:auto">
          <p style="width:max-content;padding:4px;background:var(--surface-panel)">Probe row {i}</p>
          <div style="display:flex;gap:6px;flex-wrap:wrap">
          <button type="button" data-probe="dark">Probe dark</button>
          <button type="button" data-probe="light">Probe light</button>
          <button type="button" data-probe="plain">Probe plain</button>
          <button type="button" data-probe="a">Probe pattern a</button>
          <button type="button" data-probe="b">Probe pattern b</button></div></div>''' for i in range(15))
        panel = '<section id="gesture-panel" style="min-height:6300px;background:var(--page-bg)">' + rows + '</section>'
        fixture = html.replace('<div class="mobile-bottom">',panel+'<div class="mobile-bottom">',1)
        script = '''<script>
        addEventListener('load', () => {
            const panel = document.getElementById('gesture-panel');
            panel.addEventListener('click', event => {
                const kind = event.target.dataset.probe;
                if (kind === 'light' || kind === 'dark') {
                    const appearance = document.getElementById('appearance');
                    appearance.value = kind;
                    appearance.dispatchEvent(new Event('change'));
                } else if (kind === 'plain') {
                    panel.style.background = 'var(--page-bg)';
                } else if (kind === 'a' || kind === 'b') {
                    const first = kind === 'a' ? '#e00000' : '#0040f0';
                    const second = kind === 'a' ? '#0040f0' : '#e00000';
                    panel.style.background = `repeating-linear-gradient(90deg,${first} 0 25vw,${second} 25vw 50vw)`;
                }
            });
            requestAnimationFrame(() => requestAnimationFrame(() => {
                scrollTo(0,panel.offsetTop+500);
                const ready = document.createElement('p');
                ready.textContent = 'Gesture fixture ready';
                ready.style.cssText = 'margin:0;padding:0';
                panel.children[2].prepend(ready);
            }));
        });
        </script>'''
        fixture = fixture.replace('</body>',script+'</body>',1)
        (destination/'gesture.html').write_text(fixture)
        identities['variants'][variant] = {'html_sha256':hashlib.sha256(html.encode()).hexdigest(), 'gesture_fixture_sha256':hashlib.sha256(fixture.encode()).hexdigest()}
    (EVIDENCE/'source-identities.json').write_text(json.dumps(identities,indent=2))
    for folder in ['Launcher','UITests']:
        (WORK/folder).mkdir(exist_ok=True)
    (WORK/'project.yml').write_text(PROJECT)
    (WORK/'Launcher/App.swift').write_text(LAUNCHER)
    (WORK/'UITests/SafariChromeCapture.swift').write_text(UI_TEST)
    configure('baseline','light')


def configure(variant,appearance):
    assert variant in VARIANTS and appearance in ['light','dark']
    name=f'{variant}-{appearance}'
    url=f'http://127.0.0.1:8000/{variant}/revgeo.html?gesture-probe={name}'
    (WORK/'UITests/ProbeConfig.swift').write_text(f'enum ProbeConfig {{ static let name = {json.dumps(name)}; static let url = {json.dumps(url)} }}\n')


def device():
    runtimes=json.loads(run('xcrun','simctl','list','runtimes','-j'))['runtimes']
    runtime=next(r for r in runtimes if r.get('isAvailable') and r['name']=='iOS 27.0')
    types=json.loads(run('xcrun','simctl','list','devicetypes','-j'))['devicetypes']
    device_type=next(r for r in types if r['name']=='iPhone 15 Pro')
    udid=run('xcrun','simctl','create','SRG Safari Gesture Verification',device_type['identifier'],runtime['identifier']).decode().strip()
    (EVIDENCE/'selected-device.json').write_text(json.dumps({'udid':udid,'device_type':device_type,'runtime':runtime},indent=2))
    print(udid)


def serve():
    class Handler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header('Content-Security-Policy',"style-src-elem 'self' 'unsafe-inline'")
            super().end_headers()
    (EVIDENCE/'fixture-scope.txt').write_text('Pristine page: original production HTML. Gesture fixture: ordinary document section and normal-flow test controls. No scroll listener, fixed test overlay, root background change, or production instrumentation. Actual touch gestures use XCTest coordinate drags. Optional external font CSS blocked by test-server CSP.\n')
    http.server.ThreadingHTTPServer(('127.0.0.1',8000),functools.partial(Handler,directory=str(WORK/'www'))).serve_forever()


if __name__=='__main__':
    {'prepare':prepare,'configure':lambda:configure(*sys.argv[2:]),'device':device,'serve':serve}[sys.argv[1]]()
