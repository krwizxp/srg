import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

baseline = Path(tempfile.mkdtemp(prefix="goal-browser-", dir=os.environ.get("RUNNER_TEMP")))
for name in ("revgeo.html", "icon.svg", "site.webmanifest"):
    original = subprocess.check_output(["git", "show", sys.argv[1] + ":" + name])
    (baseline / name).write_bytes(original)
    assert original == Path(name).read_bytes(), "Unchanged browser evidence no longer applies"
servers = []
try:
    for port, path in ((4173, baseline), (4174, Path.cwd())):
        servers.append(subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for _ in range(100):
        try:
            for port in (4173, 4174):
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/revgeo.html", timeout=1) as response:
                    assert response.status == 200
            break
        except OSError:
            time.sleep(0.1)
    subprocess.run(["node", ".github/validation/round2-browser.cjs"], check=True, timeout=240)
finally:
    for server in servers:
        server.terminate()
    for server in servers:
        server.wait(timeout=10)
