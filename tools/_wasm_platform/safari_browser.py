import argparse
import base64
import json
import pathlib
import platform
import urllib.error
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("url")
parser.add_argument("fixtures")
parser.add_argument("output")
parser.add_argument("--driver", default="http://127.0.0.1:4444")
parser.add_argument("--device-udid")
parser.add_argument("--device-name")
parser.add_argument("--wasm-sha256")
parser.add_argument("--boundary-path", default="./korea_boundary.js?v=6")
args = parser.parse_args()
output = pathlib.Path(args.output)
output.mkdir(parents=True, exist_ok=True)
source = pathlib.Path(__file__).with_name("browser_contract.js").read_text()
fixtures = json.loads(pathlib.Path(args.fixtures).read_text())
capabilities = {"browserName": "safari", "platformName": "ios" if args.device_udid else "mac"}
if args.device_udid:
    capabilities.update({"safari:useSimulator": True, "safari:deviceUDID": args.device_udid})
if args.device_name:
    capabilities["safari:deviceName"] = args.device_name

def request(method, endpoint, data=None):
    payload = json.dumps(data).encode() if data is not None else None
    query = urllib.request.Request(args.driver + endpoint, data=payload,
                                   headers={"Content-Type": "application/json"}, method=method)
    try:
        with urllib.request.urlopen(query, timeout=300) as response:
            body = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode()) from error
    value = body.get("value", body)
    if isinstance(value, dict) and value.get("error"):
        raise RuntimeError(json.dumps(value))
    return value

session = None
report = None
metadata = {"host": platform.platform(), "hostMachine": platform.machine(),
            "requestedCapabilities": capabilities, "physicalDevice": False,
            "environment": "Apple iOS/iPadOS Simulator Safari" if args.device_udid else "native macOS Safari"}
cleanup = []
try:
    created = request("POST", "/session", {"capabilities": {"alwaysMatch": capabilities}})
    session = created["sessionId"]
    metadata["actualCapabilities"] = created["capabilities"]
    prefix = "/session/" + session
    request("POST", prefix + "/timeouts", {"script": 240000, "pageLoad": 60000, "implicit": 0})
    request("POST", prefix + "/url", {"url": args.url})
    config = {"fixtures": fixtures, "wasm": bool(args.wasm_sha256), "wasmSha256": args.wasm_sha256,
              "wasmPath": "./coordinates.wasm", "boundaryPath": args.boundary_path}
    script = "const done = arguments[arguments.length - 1]; (" + source + ")(arguments[0]).then(done, error => done({complete:false,failure:String(error)}));"
    report = request("POST", prefix + "/execute/async", {"script": script, "args": [config]})
    screenshot = request("GET", prefix + "/screenshot")
    (output / "final-page.png").write_bytes(base64.b64decode(screenshot))
except Exception as error:
    report = {"complete": False, "failure": str(error)}
finally:
    if session:
        try:
            request("DELETE", "/session/" + session)
            cleanup.append({"webdriverSessionClosed": True})
        except Exception as error:
            cleanup.append({"webdriverSessionClosed": False, "error": str(error)})
    result = {"metadata": metadata, "report": report, "cleanup": cleanup}
    (output / "safari-browser-results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "complete": bool(report and report.get("complete")),
                      "passed": report.get("passed") if report else None,
                      "required": report.get("required") if report else None}))
    if not report or not report.get("complete") or any(not item.get("webdriverSessionClosed") for item in cleanup):
        raise SystemExit(1)
