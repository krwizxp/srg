"""One-off native Safari UI capture; never changes the served application."""
import hashlib
import functools
import http.server
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / 'safari27-work'
EVIDENCE = REPO / 'safari27-evidence'
BASELINE = '532c427a10534d55805afa8d4bea16e66d4bef22'
CANDIDATE = '1a79b6cad5faed3cb67e8505a5a22ce9ab0f86af'

PROJECT = '''name: SafariProbe
options:
  bundleIdPrefix: dev.srg.validation
  deploymentTarget:
    iOS: "27.0"
settings:
  base:
    SWIFT_VERSION: "5.0"
    CODE_SIGNING_ALLOWED: NO
    GENERATE_INFOPLIST_FILE: YES
targets:
  SafariProbe:
    type: application
    platform: iOS
    sources: [Launcher]
    settings:
      base:
        TARGETED_DEVICE_FAMILY: "1"
        INFOPLIST_KEY_UILaunchScreen_Generation: YES
  SafariProbeUITests:
    type: bundle.ui-testing
    platform: iOS
    sources: [UITests]
    dependencies:
      - target: SafariProbe
schemes:
  SafariProbe:
    build:
      targets:
        SafariProbe: all
    test:
      targets: [SafariProbeUITests]
'''

LAUNCHER = '''import SwiftUI
import UIKit

@main
struct SafariProbeApp: App {
    var body: some Scene {
        WindowGroup {
            Text("Native Safari verification launcher")
                .onAppear {
                    let args = ProcessInfo.processInfo.arguments
                    guard let index = args.firstIndex(of: "--url"),
                          args.indices.contains(index + 1),
                          let url = URL(string: args[index + 1]) else { return }
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        UIApplication.shared.open(url)
                    }
                }
        }
    }
}
'''

UI_TEST = '''import XCTest

final class SafariChromeCapture: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    func capture(_ phase: String, safari: XCUIApplication) {
        let screen = XCUIScreen.main.screenshot()
        let screenshot = XCTAttachment(screenshot: screen)
        screenshot.name = "\\(ProbeConfig.name)__\\(phase)"
        screenshot.lifetime = .keepAlways
        add(screenshot)
        let tree = XCTAttachment(string: safari.debugDescription)
        tree.name = "\\(ProbeConfig.name)__\\(phase)__accessibility"
        tree.lifetime = .keepAlways
        add(tree)

        if phase.hasPrefix("glass-") {
            let matches = safari.descendants(matching: .any)
                .matching(NSPredicate(format: "label == %@", "결과")).allElementsBoundByIndex
            let boxes = matches.filter { !$0.frame.isEmpty }.map { element -> [String: Double] in
                let rect = element.frame
                return ["x": Double(rect.minX), "y": Double(rect.minY),
                        "width": Double(rect.width), "height": Double(rect.height)]
            }
            let payload: [String: Any] = [
                "screenWidth": Double(safari.frame.width),
                "screenHeight": Double(safari.frame.height),
                "results": boxes
            ]
            if let bytes = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys]),
               let text = String(data: bytes, encoding: .utf8) {
                let attachment = XCTAttachment(string: text)
                attachment.name = "\\(ProbeConfig.name)__\\(phase)__geometry"
                attachment.lifetime = .keepAlways
                add(attachment)
            }
        }
    }

    func testNativeSafariThemeSwitch() throws {
        let safari = XCUIApplication(bundleIdentifier: "com.apple.mobilesafari")
        safari.terminate()
        let launcher = XCUIApplication()
        launcher.launchArguments = ["--url", ProbeConfig.url]
        launcher.launch()
        XCTAssertTrue(safari.wait(for: .runningForeground, timeout: 20))

        // Safari's first-launch panels belong to the native application.
        for _ in 0..<5 {
            let toggle = safari.descendants(matching: .any).matching(NSPredicate(format: "label == %@", "다크 모드")).firstMatch
            if toggle.exists { break }
            for title in ["Continue", "Start Browsing", "Not Now", "Continue Without Extensions", "Open"] {
                let button = safari.buttons[title].firstMatch
                if button.exists && button.isHittable { button.tap() }
            }
            Thread.sleep(forTimeInterval: 0.5)
        }
        capture("loaded", safari: safari)
        let toggle = safari.descendants(matching: .any).matching(NSPredicate(format: "label == %@", "다크 모드")).firstMatch
        XCTAssertTrue(toggle.waitForExistence(timeout: 30), "Theme button must be exposed by native Safari accessibility")
        XCTAssertTrue(toggle.isHittable)
        Thread.sleep(forTimeInterval: 2)
        capture("initial", safari: safari)
        toggle.tap()
        Thread.sleep(forTimeInterval: 2)
        capture("opposite-2s", safari: safari)
        Thread.sleep(forTimeInterval: 4)
        capture("opposite-6s", safari: safari)
        toggle.tap()
        Thread.sleep(forTimeInterval: 2)
        capture("returned", safari: safari)

        // Metamorphic optical check: only a test underlay changes red -> blue.
        // The resulting pixel difference inside the unselected glass capsule
        // is analyzed independently of the CSS property values.
        for shade in ["red", "blue"] {
            launcher.terminate()
            let fixtureURL = ProbeConfig.url.replacingOccurrences(
                of: "revgeo.html", with: "glass-\\(shade).html")
            launcher.launchArguments = ["--url", fixtureURL]
            launcher.launch()
            XCTAssertTrue(safari.wait(for: .runningForeground, timeout: 20))
            let marker = safari.descendants(matching: .any)
                .matching(NSPredicate(format: "label == %@", "Glass probe \\(shade)")).firstMatch
            XCTAssertTrue(marker.waitForExistence(timeout: 30))
            Thread.sleep(forTimeInterval: 2)
            capture("glass-\\(shade)", safari: safari)
        }

        // Capturing successfully is not a claim that Safari's colors passed.
        // The exported full-screen images include native UI for separate review.
    }
}
'''


def run(*args):
    return subprocess.check_output(args, cwd=REPO)


def prepare():
    WORK.mkdir(exist_ok=True)
    EVIDENCE.mkdir(exist_ok=True)
    identities = {}
    for label, commit in [('baseline', BASELINE), ('candidate', CANDIDATE)]:
        destination = WORK / 'www' / label
        destination.mkdir(parents=True, exist_ok=True)
        # Every path is read from the pinned Git tree, including relative assets.
        paths = run('git', 'ls-tree', '-r', '--name-only', commit).decode().splitlines()
        for path in paths:
            if path == 'revgeo.html' or path == 'site.webmanifest' or path.startswith('assets/') or path.endswith(('.svg', '.png', '.ico')):
                target = destination / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(run('git', 'show', f'{commit}:{path}'))
        html = (destination / 'revgeo.html').read_bytes()
        identities[label] = {'commit': commit, 'html_sha256': hashlib.sha256(html).hexdigest()}
        # These two explicit test-only documents add a colored underlay behind
        # the unchanged navigation CSS. Pristine production pages above are
        # used for all native chrome assertions.
        identities[label]['fixtures'] = {}
        for shade, color in [('red', '#e00000'), ('blue', '#0040f0')]:
            marker = (
                f'<div id="glass-probe" role="img" aria-label="Glass probe {shade}" '
                f'style="position:fixed;left:0;right:0;bottom:12px;height:90px;'
                f'z-index:24;pointer-events:none;background:{color}"></div>'
            )
            fixture = html.decode().replace('<div class="mobile-bottom">',
                marker + '<div class="mobile-bottom">', 1)
            fixture_bytes = fixture.encode()
            (destination / f'glass-{shade}.html').write_bytes(fixture_bytes)
            identities[label]['fixtures'][shade] = hashlib.sha256(fixture_bytes).hexdigest()
    assert identities['candidate']['html_sha256'] == '1f3eb95b13f4d7cf2b40461bbd70496c15165f871f27c603a109a6cd2ca22852'
    (EVIDENCE / 'source-identities.json').write_text(json.dumps(identities, indent=2))
    for directory in ['Launcher', 'UITests']:
        (WORK / directory).mkdir(exist_ok=True)
    (WORK / 'project.yml').write_text(PROJECT)
    (WORK / 'Launcher/App.swift').write_text(LAUNCHER)
    (WORK / 'UITests/SafariChromeCapture.swift').write_text(UI_TEST)
    configure('candidate', 'dark')


def configure(variant, appearance):
    assert variant in ['baseline', 'candidate']
    assert appearance in ['light', 'dark']
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
    udid = run('xcrun', 'simctl', 'create', 'SRG Safari 27 Verification', device_type['identifier'], runtime['identifier']).decode().strip()
    (EVIDENCE / 'selected-device.json').write_text(json.dumps({'udid': udid, 'device_type': device_type, 'runtime': runtime}, indent=2))
    print(udid)


def serve():
    # The optional CDN font stylesheet stalled first paint on the runner.
    # Exercise the application's existing system-font fallback without changing
    # any document bytes or disabling TLS verification/browser security.
    policy = "style-src-elem 'self' 'unsafe-inline'"

    class Handler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header('Content-Security-Policy', policy)
            super().end_headers()

    (EVIDENCE / 'network-fixture.txt').write_text(
        'Native Safari UI check with optional external font CSS blocked.\n'
        'Original HTML/inline styles/scripts/assets remain byte-identical.\n'
        f'Test-only HTTP response header: Content-Security-Policy: {policy}\n')
    handler = functools.partial(Handler, directory=str(WORK / 'www'))
    http.server.ThreadingHTTPServer(('127.0.0.1', 8000), handler).serve_forever()


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        prepare()
    elif sys.argv[1] == 'configure':
        configure(*sys.argv[2:])
    elif sys.argv[1] == 'device':
        device()
    elif sys.argv[1] == 'serve':
        serve()
    else:
        raise SystemExit('Unknown operation')
