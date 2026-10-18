"""Temporary native Safari chrome transmission experiment, not product code."""
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
BASELINE = '66e5cbced16f43bb7beea2656e82eeb7238e6c34'
VARIANTS = ['baseline', 'clear', 'inset']

PROJECT = 'name: SafariProbe\noptions:\n  bundleIdPrefix: dev.srg.validation\n  deploymentTarget:\n    iOS: "27.0"\nsettings:\n  base:\n    SWIFT_VERSION: "5.0"\n    CODE_SIGNING_ALLOWED: NO\n    GENERATE_INFOPLIST_FILE: YES\ntargets:\n  SafariProbe:\n    type: application\n    platform: iOS\n    sources: [Launcher]\n    settings:\n      base:\n        TARGETED_DEVICE_FAMILY: "1"\n        INFOPLIST_KEY_UILaunchScreen_Generation: YES\n  SafariProbeUITests:\n    type: bundle.ui-testing\n    platform: iOS\n    sources: [UITests]\n    dependencies:\n      - target: SafariProbe\nschemes:\n  SafariProbe:\n    build:\n      targets:\n        SafariProbe: all\n    test:\n      targets: [SafariProbeUITests]\n'

LAUNCHER = 'import SwiftUI\nimport UIKit\n\n@main\nstruct SafariProbeApp: App {\n    var body: some Scene {\n        WindowGroup {\n            Text("Native Safari verification launcher")\n                .onAppear {\n                    let args = ProcessInfo.processInfo.arguments\n                    guard let index = args.firstIndex(of: "--url"),\n                          args.indices.contains(index + 1),\n                          let url = URL(string: args[index + 1]) else { return }\n                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {\n                        UIApplication.shared.open(url)\n                    }\n                }\n        }\n    }\n}\n'
UI_TEST = r'''import XCTest

final class SafariTransmissionCapture: XCTestCase {
    override func setUpWithError() throws { continueAfterFailure = false }

    func capture(_ phase: String, safari: XCUIApplication) {
        let shot = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
        shot.name = "\(ProbeConfig.name)__\(phase)"
        shot.lifetime = .keepAlways
        add(shot)
        let tree = XCTAttachment(string: safari.debugDescription)
        tree.name = "\(ProbeConfig.name)__\(phase)__accessibility"
        tree.lifetime = .keepAlways
        add(tree)
    }

    func open(_ url: String, safari: XCUIApplication) {
        let launcher = XCUIApplication()
        launcher.terminate()
        launcher.launchArguments = ["--url", url]
        launcher.launch()
        XCTAssertTrue(safari.wait(for: .runningForeground, timeout: 20))
    }

    func testNativeSafariTransmissionAndTheme() throws {
        let safari = XCUIApplication(bundleIdentifier: "com.apple.mobilesafari")
        safari.terminate()
        open(ProbeConfig.url, safari: safari)
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
        XCTAssertTrue(toggle.isHittable)
        Thread.sleep(forTimeInterval: 2)
        capture("initial", safari: safari)
        toggle.tap()
        Thread.sleep(forTimeInterval: 2)
        capture("opposite", safari: safari)
        toggle.tap()
        Thread.sleep(forTimeInterval: 2)
        capture("returned", safari: safari)

        for theme in ["light", "dark"] {
            for pattern in ["a", "b"] {
                let fixture = ProbeConfig.url.replacingOccurrences(of: "revgeo.html", with: "transmission-\(theme)-\(pattern).html")
                open(fixture, safari: safari)
                let label = "Transmission ready \(theme) \(pattern)"
                let marker = safari.descendants(matching: .any).matching(NSPredicate(format: "label == %@", label)).firstMatch
                XCTAssertTrue(marker.waitForExistence(timeout: 30), label)
                Thread.sleep(forTimeInterval: 2)
                capture("transmission-\(theme)-\(pattern)", safari: safari)
            }
        }
    }
}
'''


def run(*args):
    return subprocess.check_output(args, cwd=REPO)


def apply_variant(source, variant):
    if variant == 'baseline':
        return source
    tint = ('        /* An explicit faint tint refreshes Safari edge colors without covering the glass. */\n'
            '        background-color: color-mix(in srgb, var(--page-bg) 1%, transparent);\n')
    fade = '      .mobile-bottom::before { content: ""; position: absolute; inset: -18px 0 0; z-index: -1; pointer-events: none; background: linear-gradient(transparent, var(--page-bg)); }\n'
    assert source.count(tint) == source.count(fade) == 1
    candidate = source.replace(tint, '        /* Keep the outer dock clear so Safari can show scrolling content below it. */\n').replace(fade, '')
    if variant == 'inset':
        anchor = 'position: fixed; z-index: 25; bottom: 0; left: 0; right: 0; display: flex;'
        assert candidate.count(anchor) == 1
        candidate = candidate.replace(anchor, 'position: fixed; z-index: 25; bottom: 8px; left: 0; right: 0; display: flex;')
    return candidate


def prepare():
    WORK.mkdir(exist_ok=True)
    EVIDENCE.mkdir(exist_ok=True)
    source = run('git', 'show', f'{BASELINE}:revgeo.html').decode()
    assert hashlib.sha256(source.encode()).hexdigest() == '1f3eb95b13f4d7cf2b40461bbd70496c15165f871f27c603a109a6cd2ca22852'
    identities = {'baseline_commit': BASELINE, 'variants': {}, 'thresholds': {
        'native_pattern_max_channel_difference_min': 24,
        'native_pattern_left_right_red_blue_contrast_min': 16,
        'native_pristine_theme_max_channel_error': 40,
        'native_edge_roi_x': [[0.005, 0.02], [0.98, 0.995]],
        'native_edge_roi_y': [0.92, 0.965]}}
    for variant in VARIANTS:
        destination = WORK / 'www' / variant
        destination.mkdir(parents=True, exist_ok=True)
        for path in ['icon.svg', 'site.webmanifest']:
            (destination / path).write_bytes(run('git', 'show', f'{BASELINE}:{path}'))
        html = apply_variant(source, variant)
        (destination / 'revgeo.html').write_text(html)
        identity = {'html_sha256': hashlib.sha256(html.encode()).hexdigest(), 'fixtures': {}}
        for theme in ['light', 'dark']:
            for pattern in ['a', 'b']:
                first, second = ('#e00000', '#0040f0') if pattern == 'a' else ('#0040f0', '#e00000')
                # Ordinary document content: no fixed/sticky positioning, no changes
                # to the product dock, root or body backgrounds, or viewport rules.
                panel = (f'<section id="transmission-panel" style="height:6000px;padding-top:1250px;'
                         f'background:repeating-linear-gradient(90deg,{first} 0 25vw,{second} 25vw 50vw)">'
                         '<p id="transmission-ready" style="width:max-content;max-width:90%;padding:8px;background:white;color:black">Pending transmission</p></section>')
                fixture = html.replace('<div class="mobile-bottom">', panel + '<div class="mobile-bottom">', 1)
                script = f'''<script>
                addEventListener('load', () => {{
                    const appearance = document.getElementById('appearance');
                    appearance.value = {json.dumps(theme)};
                    appearance.dispatchEvent(new Event('change'));
                    requestAnimationFrame(() => requestAnimationFrame(() => {{
                        const panel = document.getElementById('transmission-panel');
                        scrollTo(0, panel.offsetTop + 1200);
                        requestAnimationFrame(() => {{
                            document.getElementById('transmission-ready').textContent = 'Transmission ready {theme} {pattern}';
                        }});
                    }}));
                }});
                </script>'''
                fixture = fixture.replace('</body>', script + '</body>', 1)
                name = f'transmission-{theme}-{pattern}.html'
                (destination / name).write_text(fixture)
                identity['fixtures'][name] = hashlib.sha256(fixture.encode()).hexdigest()
        identities['variants'][variant] = identity
    (EVIDENCE / 'source-identities.json').write_text(json.dumps(identities, indent=2))
    for folder in ['Launcher', 'UITests']:
        (WORK / folder).mkdir(exist_ok=True)
    (WORK / 'project.yml').write_text(PROJECT)
    (WORK / 'Launcher/App.swift').write_text(LAUNCHER)
    (WORK / 'UITests/SafariChromeCapture.swift').write_text(UI_TEST)
    configure('clear', 'light')


def configure(variant, appearance):
    assert variant in VARIANTS and appearance in ['light', 'dark']
    name = f'{variant}-{appearance}'
    url = f'http://127.0.0.1:8000/{variant}/revgeo.html?native-probe={name}'
    (WORK / 'UITests/ProbeConfig.swift').write_text(
        f'enum ProbeConfig {{ static let name = {json.dumps(name)}; static let url = {json.dumps(url)} }}\n')


def device():
    runtimes = json.loads(run('xcrun', 'simctl', 'list', 'runtimes', '-j'))['runtimes']
    runtime = next(row for row in runtimes if row.get('isAvailable') and row['name'] == 'iOS 27.0')
    types = json.loads(run('xcrun', 'simctl', 'list', 'devicetypes', '-j'))['devicetypes']
    preferred = ['iPhone 15 Pro', 'iPhone 17', 'iPhone 18 Pro']
    device_type = next(row for name in preferred for row in types if row['name'] == name)
    udid = run('xcrun', 'simctl', 'create', 'SRG Safari Transmission Verification', device_type['identifier'], runtime['identifier']).decode().strip()
    (EVIDENCE / 'selected-device.json').write_text(json.dumps({'udid': udid, 'device_type': device_type, 'runtime': runtime}, indent=2))
    print(udid)


def serve():
    class Handler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header('Content-Security-Policy', "style-src-elem 'self' 'unsafe-inline'")
            super().end_headers()
    (EVIDENCE / 'network-fixture.txt').write_text('Optional external font CSS blocked using test-server CSP; existing system-font fallback. Inline application scripts and browser security unchanged. Transmission pages add a normal-flow colored section and test-only readiness/scroll/appearance script. Pristine pages remain byte-identical to the pinned variant.\n')
    http.server.ThreadingHTTPServer(('127.0.0.1', 8000), functools.partial(Handler, directory=str(WORK / 'www'))).serve_forever()


if __name__ == '__main__':
    {'prepare': prepare, 'configure': lambda: configure(*sys.argv[2:]), 'device': device, 'serve': serve}[sys.argv[1]]()
