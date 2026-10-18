#!/bin/bash
set -euo pipefail
variant="$1"
appearance="$2"
name="$variant-$appearance"
python3 .github/safari-gesture.py serve > "safari27-evidence/http-$name.log" 2>&1 &
PROBE_SERVER_PID=$!
trap 'kill "$PROBE_SERVER_PID"' EXIT
for attempt in 1 2 3 4 5; do
  if curl --fail --silent http://127.0.0.1:8000/baseline/revgeo.html > /dev/null; then break; fi
  sleep 1
done
xcrun simctl ui "$PROBE_UDID" appearance "$appearance"
python3 .github/safari-gesture.py configure "$variant" "$appearance"
result=0
if ! xcodebuild test -project safari27-work/SafariProbe.xcodeproj -scheme SafariProbe \
  -destination "id=$PROBE_UDID" -parallel-testing-enabled NO \
  -collect-test-diagnostics never \
  -derivedDataPath safari27-work/DerivedData \
  -resultBundlePath "safari27-evidence/$name.xcresult" \
  CODE_SIGNING_ALLOWED=NO > "safari27-evidence/$name.log" 2>&1; then
  result=1
fi
if ! xcrun xcresulttool export attachments --path "safari27-evidence/$name.xcresult" \
  --output-path "safari27-evidence/$name-attachments"; then
  result=1
fi
tail -n 60 "safari27-evidence/$name.log"
exit "$result"
