#!/usr/bin/env bash
set -euo pipefail

action="${SRG_ACTION:-}"
artifact_prefix="${SRG_ARTIFACT_NAME:-srg-result}"
artifact_prefix="$(
  printf '%s' "$artifact_prefix" |
    tr '\r\n' '--' |
    sed \
      -e 's/^[[:space:]]*//' \
      -e 's/[[:space:]]*$//' \
      -e 's/[[:cntrl:]\\/:*?"<>|]/-/g'
)"
if [[ -z "${artifact_prefix//[[:space:]]/}" ]]; then
  artifact_prefix="srg-result"
fi
artifact_prefix="${artifact_prefix:0:80}"

artifact_dir="$PWD/artifacts"
exe_path="$PWD/target/release/srg"
log_path="$artifact_dir/${artifact_prefix}-console.log"
data_path="$PWD/random_data.txt"
copied_data_path="$artifact_dir/${artifact_prefix}-random_data.txt"
github_output="${GITHUB_OUTPUT:-/dev/null}"

mkdir -p "$artifact_dir"
rm -f "$data_path"
: > "$log_path"
printf 'artifact_name_prefix=%s\n' "$artifact_prefix" >> "$github_output"
printf 'log_artifact_path=%s\n' "$log_path" >> "$github_output"

error_logged=0

log_error() {
  local message="$1"
  {
    printf '\n[workflow-error]\n'
    printf '%s\n' "$message"
  } >> "$log_path"
  error_logged=1
}

die() {
  local message="$1"
  log_error "$message"
  printf '%s\n' "$message" >&2
  exit 1
}

trap 'status=$?; if (( status != 0 && error_logged == 0 )); then log_error "Workflow failed with exit code $status."; fi' EXIT

if [[ ! -x "$exe_path" ]]; then
  die "Built SRG binary not found or not executable: $exe_path"
fi

python_bin=""
if command -v python3 >/dev/null 2>&1 &&
  python3 -c 'import sys; raise SystemExit(0 if sys.version_info[0] == 3 else 1)' >/dev/null 2>&1; then
  python_bin="python3"
elif command -v python >/dev/null 2>&1 && python -c 'import sys; raise SystemExit(0 if sys.version_info[0] == 3 else 1)' >/dev/null 2>&1; then
  python_bin="python"
else
  die "Python 3 is required to validate numeric workflow inputs."
fi

trim_string() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

require_non_empty() {
  local value="${1:-}"
  local name="$2"
  if [[ -z "${value//[[:space:]]/}" ]]; then
    die "$name is required for action '$action'."
  fi
}

split_ladder_input() {
  local value="${1:-}"
  local name="$2"
  local output_name="$3"
  local -n output="$output_name"
  local raw_entries=()

  require_non_empty "$value" "$name"
  if [[ "$value" == *$'\r'* || "$value" == *$'\n'* ]]; then
    die "$name must be a single comma-separated line."
  fi
  if [[ "$value" == *, ]]; then
    die "$name entries must not be empty."
  fi

  IFS=',' read -r -a raw_entries <<< "$value"
  output=()
  for raw_entry in "${raw_entries[@]}"; do
    local entry
    entry="$(trim_string "$raw_entry")"
    if [[ -z "${entry//[[:space:]]/}" ]]; then
      die "$name entries must not be empty."
    fi
    output+=("$entry")
  done
}

join_by_comma() {
  local IFS=,
  printf '%s' "$*"
}

parse_int_range() {
  local value="${1:-}"
  local name="$2"
  local min_value="$3"
  local max_value="$4"
  local invalid_message="$5"
  local output

  if ! output="$("$python_bin" - "$value" "$name" "$min_value" "$max_value" "$invalid_message" 2>&1 <<'PY'
import sys

raw, name, min_raw, max_raw, invalid_message = sys.argv[1:]
raw = raw.strip()
try:
    value = int(raw, 10)
except ValueError:
    raise SystemExit(invalid_message)

minimum = int(min_raw, 10)
maximum = int(max_raw, 10)
if value < minimum or value > maximum:
    raise SystemExit(f"{name} must be between {minimum} and {maximum}.")

print(value, end="")
PY
  )"; then
    die "$output"
  fi

  printf '%s' "$output"
}

parse_float() {
  local value="${1:-}"
  local name="$2"
  local output

  if ! output="$("$python_bin" - "$value" "$name" 2>&1 <<'PY'
import math
import sys

raw, name = sys.argv[1:]
try:
    value = float(raw.strip())
except ValueError:
    raise SystemExit(f"{name} must be a valid floating-point number.")

if not math.isfinite(value):
    raise SystemExit(f"{name} must be finite.")
if value != 0.0 and abs(value) < sys.float_info.min:
    raise SystemExit(f"{name} must not be subnormal.")

print(format(value, ".17g"), end="")
PY
  )"; then
    die "$output"
  fi

  printf '%s' "$output"
}

ensure_float_order() {
  local min_value="$1"
  local max_value="$2"

  if ! "$python_bin" - "$min_value" "$max_value" >/dev/null 2>&1 <<'PY'
import sys

minimum, maximum = (float(value) for value in sys.argv[1:])
if maximum < minimum:
    raise SystemExit(1)
PY
  then
    die "float_max must be greater than or equal to float_min."
  fi
}

write_process_log() {
  local stdout_path="$1"
  local stderr_path="$2"

  cat "$stdout_path" > "$log_path"
  if [[ -s "$stderr_path" ]]; then
    {
      printf '\n[stderr]\n'
      cat "$stderr_path"
    } >> "$log_path"
  fi
}

run_srg_with_lines() {
  local stdout_path
  local stderr_path
  local input_path
  local exit_code=0

  stdout_path="$(mktemp)"
  stderr_path="$(mktemp)"
  input_path="$(mktemp)"
  printf '%s\n' "$@" > "$input_path"

  if "$exe_path" < "$input_path" > "$stdout_path" 2> "$stderr_path"; then
    exit_code=0
  else
    exit_code=$?
  fi
  write_process_log "$stdout_path" "$stderr_path"
  rm -f "$stdout_path" "$stderr_path" "$input_path"

  if (( exit_code != 0 )); then
    die "srg exited with code $exit_code. See $log_path."
  fi
}

run_time_sync_observe() {
  local time_host="$1"
  local observe_seconds="$2"
  local stdout_path
  local stderr_path
  local exit_code=0

  stdout_path="$(mktemp)"
  stderr_path="$(mktemp)"

  if {
    printf '6\n'
    printf '5\n'
    printf '%s\n' "$time_host"
    printf '\n'
    sleep "$observe_seconds"
    printf '\n'
    printf 'q\n'
  } | "$exe_path" > "$stdout_path" 2> "$stderr_path"; then
    exit_code=0
  else
    exit_code=$?
  fi
  write_process_log "$stdout_path" "$stderr_path"
  rm -f "$stdout_path" "$stderr_path"

  if (( exit_code != 0 )); then
    die "srg exited with code $exit_code. See $log_path."
  fi
}

case "$action" in
  generate-single)
    run_srg_with_lines "6" "3" "q"
    ;;
  generate-multiple)
    count="$(
      parse_int_range \
        "${SRG_COUNT:-}" \
        "count" \
        "1" \
        "100000" \
        "count must be a valid unsigned integer."
    )"
    run_srg_with_lines "6" "4" "$count" "q"
    ;;
  ladder)
    players=()
    results=()
    split_ladder_input "${SRG_PLAYERS:-}" "players" players
    split_ladder_input "${SRG_RESULTS:-}" "results" results
    if (( ${#players[@]} < 2 || ${#players[@]} > 512 )); then
      die "players must include between 2 and 512 entries."
    fi
    if (( ${#results[@]} != ${#players[@]} )); then
      die "results entry count must match players entry count."
    fi
    run_srg_with_lines "6" "1" "$(join_by_comma "${players[@]}")" "$(join_by_comma "${results[@]}")" "q"
    ;;
  random-integer)
    min="$(
      parse_int_range \
        "${SRG_INT_MIN:-}" \
        "int_min" \
        "-9223372036854775808" \
        "9223372036854775807" \
        "int_min must be a valid integer."
    )"
    max="$(
      parse_int_range \
        "${SRG_INT_MAX:-}" \
        "int_max" \
        "-9223372036854775808" \
        "9223372036854775807" \
        "int_max must be a valid integer."
    )"
    if (( max < min )); then
      die "int_max must be greater than or equal to int_min."
    fi
    run_srg_with_lines "6" "2" "1" "$min" "$max" "q"
    ;;
  random-float)
    min="$(parse_float "${SRG_FLOAT_MIN:-}" "float_min")"
    max="$(parse_float "${SRG_FLOAT_MAX:-}" "float_max")"
    ensure_float_order "$min" "$max"
    run_srg_with_lines "6" "2" "2" "$min" "$max" "q"
    ;;
  time-sync-observe)
    time_host="${SRG_TIME_HOST:-}"
    observe_seconds="$(
      parse_int_range \
        "${SRG_OBSERVE_SECONDS:-}" \
        "observe_seconds" \
        "1" \
        "60" \
        "observe_seconds must be a valid integer."
    )"
    require_non_empty "$time_host" "time_host"
    run_time_sync_observe "$time_host" "$observe_seconds"
    ;;
  *)
    die "Unsupported action: $action"
    ;;
esac

if [[ -f "$data_path" ]]; then
  cp -f "$data_path" "$copied_data_path"
  printf 'data_artifact_path=%s\n' "$copied_data_path" >> "$github_output"
fi
