import argparse
import functools
import hashlib
import http.server
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("platform", choices=["android", "apple", "desktop"])
parser.add_argument("product_root")
parser.add_argument("fixtures")
parser.add_argument("output")
parser.add_argument("--wasm-sha256")
parser.add_argument("--boundary-path", default="./korea_boundary.js?v=6")
parser.add_argument("--engine", default="chromium")
parser.add_argument("--channel")
parser.add_argument("--android-image", default="system-images;android-37.0;google_apis;x86_64")
parser.add_argument("--ios-runtime", default="com.apple.CoreSimulator.SimRuntime.iOS-26-5")
parser.add_argument("--apple-scope", choices=["all", "macos-safari", "ios-iphone", "ipados-ipad"], default="all")
parser.add_argument("--port", type=int, default=18974)
args = parser.parse_args()
root = pathlib.Path(args.product_root).resolve()
output = pathlib.Path(args.output).resolve()
output.mkdir(parents=True, exist_ok=True)
here = pathlib.Path(__file__).resolve().parent
env = os.environ.copy()
commands = []
cleanup = []
results = []
owned_processes = []
owned_devices = []
owned_avd = False
owned_android_emulator = False
server = None
failure = None
android_adb = None
android_avdmanager = None
serial = "emulator-5554"
counter = 0

def execute(name, command, *, check=True, timeout=120, input_text=None):
    global counter
    counter += 1
    stem = output / (str(counter).zfill(3) + "-" + name)
    with stem.with_suffix(".stdout.log").open("w") as stdout, stem.with_suffix(".stderr.log").open("w") as stderr:
        completed = subprocess.run(command, env=env, input=input_text, text=True,
                                   stdout=stdout, stderr=stderr, timeout=timeout)
    item = {"name": name, "command": [str(item) for item in command], "exitCode": completed.returncode,
            "stdout": stem.with_suffix(".stdout.log").name, "stderr": stem.with_suffix(".stderr.log").name}
    commands.append(item)
    print(json.dumps(item), flush=True)
    if check and completed.returncode:
        raise RuntimeError(name + " failed, exit " + str(completed.returncode))
    return completed.returncode, stem.with_suffix(".stdout.log").read_text()

def start(name, command):
    log = (output / (name + ".log")).open("w")
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    owned_processes.append((name, process, log))
    return process

def wait_for(name, operation, timeout):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            if operation():
                return
        except Exception as error:
            last = str(error)
        time.sleep(1)
    raise RuntimeError(name + " timed out; last error: " + str(last))

class ProductHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".wasm": "application/octet-stream", ".js": "text/javascript; charset=utf-8"}

    def log_message(self, format_string, *values):
        with (output / "http.log").open("a") as log:
            log.write(format_string % values + "\n")

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

def browser_arguments(folder):
    return ["http://127.0.0.1:" + str(args.port) + "/revgeo.html", str(pathlib.Path(args.fixtures).resolve()),
            str(output / folder), args.wasm_sha256 or "", args.boundary_path]

def run_safari(folder, udid=None, device_name=None):
    arguments = browser_arguments(folder)
    command = [sys.executable, str(here / "safari_browser.py"), *arguments[:3],
               "--boundary-path", args.boundary_path]
    if args.wasm_sha256:
        command += ["--wasm-sha256", args.wasm_sha256]
    if udid:
        command += ["--device-udid", udid, "--device-name", device_name]
    code, _ = execute(folder, command, check=False, timeout=420)
    results.append({"name": folder, "exitCode": code})
    return code

try:
    sources = {}
    for name in ["revgeo.html", "korea_boundary.js", "coordinates_wasm.js", "coordinates.wasm",
                 "data/korea-boundary-v4/tile-select-v5.bin"]:
        path = root / name
        if path.exists():
            sources[name] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if args.wasm_sha256 and sources.get("coordinates.wasm", {}).get("sha256") != args.wasm_sha256:
        raise RuntimeError("WASM source identity mismatch before serving")
    (output / "source-identity.json").write_text(json.dumps(sources, indent=2) + "\n")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), functools.partial(ProductHandler, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    if args.platform == "desktop":
        command = [shutil.which("node") or "node", str(here / "desktop_browser.mjs"),
                   *browser_arguments("desktop"), args.engine]
        if args.channel:
            command.append(args.channel)
        code, _ = execute("desktop-browser", command, check=False, timeout=420)
        results.append({"name": "desktop", "exitCode": code})
    elif args.platform == "android":
        if platform.system() != "Linux" or not pathlib.Path("/dev/kvm").exists():
            raise RuntimeError("Android hosted validation requires a Linux KVM runner")
        if not os.access("/dev/kvm", os.R_OK | os.W_OK):
            raise RuntimeError("Task runner account cannot access /dev/kvm; grant device access on this disposable runner")
        sdk = pathlib.Path(env.get("ANDROID_HOME") or env.get("ANDROID_SDK_ROOT") or "/usr/local/lib/android/sdk")
        sdkmanager = sdk / "cmdline-tools/latest/bin/sdkmanager"
        android_avdmanager = sdk / "cmdline-tools/latest/bin/avdmanager"
        android_adb = sdk / "platform-tools/adb"
        env["ANDROID_AVD_HOME"] = str(output / "avd")
        pathlib.Path(env["ANDROID_AVD_HOME"]).mkdir(parents=True, exist_ok=True)
        env["SRG_ADB"] = str(android_adb)
        env["PATH"] = str(sdk / "platform-tools") + os.pathsep + str(sdk / "emulator") + os.pathsep + env["PATH"]
        execute("sdk-licenses", [sdkmanager, "--licenses"], timeout=180, input_text="y\n" * 100)
        execute("sdk-install", [sdkmanager, "emulator", "platform-tools", args.android_image], timeout=1200, input_text="y\n" * 100)
        execute("emulator-linked-libraries", ["ldd", sdk / "emulator/qemu/linux-x86_64/qemu-system-x86_64"], check=False)
        execute("emulator-version", [sdk / "emulator/emulator", "-version"])
        execute("adb-version", [android_adb, "version"])
        _, existing = execute("adb-devices-before", [android_adb, "devices"])
        if re.search(r"^\S+\s+(device|offline|unauthorized)$", existing, re.MULTILINE):
            raise RuntimeError("Refusing to share a preexisting Android device")
        execute("create-avd", [android_avdmanager, "create", "avd", "--name", "srg_wasm_validation",
                               "--package", args.android_image, "--device", "pixel_7"], input_text="no\n")
        owned_avd = True
        start("emulator", [sdk / "emulator/emulator", "-avd", "srg_wasm_validation", "-no-window", "-no-audio",
                           "-no-snapshot", "-no-boot-anim", "-gpu", "software", "-accel", "on", "-memory", "3072",
                           "-cores", "2", "-port", "5554"])
        owned_android_emulator = True
        def android_booted():
            completed = subprocess.run([android_adb, "-s", serial, "shell", "getprop", "sys.boot_completed"],
                                       env=env, capture_output=True, text=True, timeout=10)
            return completed.returncode == 0 and completed.stdout.strip() == "1"
        wait_for("Android boot", android_booted, 480)
        execute("android-root", [android_adb, "-s", serial, "root"])
        execute("android-root-ready", [android_adb, "-s", serial, "wait-for-device"])
        execute("android-awake", [android_adb, "-s", serial, "shell", "input", "keyevent", "KEYCODE_WAKEUP"])
        execute("android-stay-awake", [android_adb, "-s", serial, "shell", "settings", "put", "global", "stay_on_while_plugged_in", "3"])
        execute("android-chrome-package", [android_adb, "-s", serial, "shell", "dumpsys", "package", "com.android.chrome"])
        execute("android-connectivity-before", [android_adb, "-s", serial, "shell", "dumpsys", "connectivity"])
        execute("android-wifi-before", [android_adb, "-s", serial, "shell", "cmd", "wifi", "status"])
        execute("android-reverse", [android_adb, "-s", serial, "reverse", "tcp:" + str(args.port), "tcp:" + str(args.port)])
        code, _ = execute("android-browser", [shutil.which("node") or "node", str(here / "android_browser.mjs"),
                                              *browser_arguments("android")], check=False, timeout=420)
        results.append({"name": "android", "exitCode": code})
        execute("android-connectivity-after", [android_adb, "-s", serial, "shell", "dumpsys", "connectivity"])
    else:
        if platform.system() != "Darwin":
            raise RuntimeError("Apple Simulator validation requires macOS and Xcode")
        execute("host-version", ["sw_vers"])
        execute("host-architecture", ["uname", "-m"])
        execute("xcode-version", ["xcodebuild", "-version"])
        _, runtime_text = execute("simulator-runtimes", ["xcrun", "simctl", "list", "runtimes", "--json"])
        runtimes = json.loads(runtime_text)["runtimes"]
        runtime = next((item for item in runtimes if item["identifier"] == args.ios_runtime and item.get("isAvailable")), None)
        if runtime is None:
            raise RuntimeError("Required simulator runtime is unavailable: " + args.ios_runtime)
        _, type_text = execute("simulator-device-types", ["xcrun", "simctl", "list", "devicetypes", "--json"])
        types = json.loads(type_text)["devicetypes"]
        execute("safaridriver-version", ["/usr/bin/safaridriver", "--version"])
        execute("enable-safaridriver", ["sudo", "-n", "/usr/bin/safaridriver", "--enable"])
        start("safaridriver", ["/usr/bin/safaridriver", "--port", "4444"])
        def driver_ready():
            with urllib.request.urlopen("http://127.0.0.1:4444/status", timeout=5) as response:
                return response.status == 200
        wait_for("SafariDriver", driver_ready, 30)
        if args.apple_scope in ["all", "macos-safari"]:
            execute("open-desktop-safari", ["open", "-a", "Safari"])
            run_safari("macos-safari")
        for name, expected in [("ios-iphone", "iPhone 17"), ("ipados-ipad", "iPad (A16)")]:
            if args.apple_scope not in ["all", name]:
                continue
            device_type = next((item for item in types if item["name"] == expected), None)
            if device_type is None:
                raise RuntimeError("Required device type unavailable: " + expected)
            _, created = execute("create-" + name, ["xcrun", "simctl", "create", "SRG-WASM-" + name,
                                                    device_type["identifier"], runtime["identifier"]])
            udid = created.strip()
            owned_devices.append(udid)
            execute("boot-" + name, ["xcrun", "simctl", "boot", udid])
            execute("boot-ready-" + name, ["xcrun", "simctl", "bootstatus", udid, "-b"], timeout=300)
            execute("open-" + name, ["open", "-a", "Simulator", "--args", "-CurrentDeviceUDID", udid])
            execute("launch-safari-" + name, ["xcrun", "simctl", "launch", udid, "com.apple.mobilesafari"])
            if run_safari(name, udid, "SRG-WASM-" + name):
                execute("failure-screenshot-" + name, ["xcrun", "simctl", "io", udid, "screenshot",
                        str(output / (name + "-setup-failure.png"))], check=False)
                execute("failure-system-log-" + name, ["xcrun", "simctl", "spawn", udid, "log", "show",
                        "--last", "5m", "--style", "compact", "--predicate",
                        'process == "MobileSafari" OR process == "webinspectord"'], check=False, timeout=60)
            execute("shutdown-" + name, ["xcrun", "simctl", "shutdown", udid])
except Exception as error:
    failure = str(error)
finally:
    if owned_android_emulator:
        try:
            execute("android-remove-reverse", [android_adb, "-s", serial, "reverse", "--remove", "tcp:" + str(args.port)], check=False)
            execute("android-stop-emulator", [android_adb, "-s", serial, "emu", "kill"], check=False)
        except Exception as error:
            cleanup.append({"androidStopError": str(error)})
    for name, process, log in reversed(owned_processes):
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            cleanup.append({"process": name, "stopped": process.poll() is not None})
        except Exception as error:
            cleanup.append({"process": name, "stopped": False, "error": str(error)})
        log.close()
    if owned_avd:
        try:
            execute("delete-task-avd", [android_avdmanager, "delete", "avd", "--name", "srg_wasm_validation"])
            cleanup.append({"taskAvdDeleted": True})
        except Exception as error:
            cleanup.append({"taskAvdDeleted": False, "error": str(error)})
    for udid in owned_devices:
        try:
            execute("cleanup-shutdown-simulator", ["xcrun", "simctl", "shutdown", udid], check=False)
            execute("cleanup-delete-simulator", ["xcrun", "simctl", "delete", udid])
            cleanup.append({"simulator": udid, "deleted": True})
        except Exception as error:
            cleanup.append({"simulator": udid, "deleted": False, "error": str(error)})
    if server:
        server.shutdown()
        server.server_close()
        cleanup.append({"httpServerClosed": True})
    cleanup_failed = any(item.get("stopped") is False or item.get("deleted") is False
                         or item.get("taskAvdDeleted") is False or "androidStopError" in item for item in cleanup)
    required_runs = 3 if args.platform == "apple" and args.apple_scope == "all" else 1
    complete = not failure and not cleanup_failed and len(results) == required_runs and all(item["exitCode"] == 0 for item in results)
    summary = {"platform": args.platform, "host": platform.platform(), "machine": platform.machine(),
               "complete": complete, "failure": failure, "requiredRuns": required_runs,
               "appleScope": args.apple_scope if args.platform == "apple" else None,
               "results": results, "commands": commands, "cleanup": cleanup,
               "wasmMime": "application/octet-stream", "crossOriginIsolationHeaders": False}
    (output / "platform-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"complete": complete, "summary": str(output / "platform-summary.json"), "failure": failure}), flush=True)
    if not complete:
        raise SystemExit(1)
