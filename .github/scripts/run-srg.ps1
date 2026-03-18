$ErrorActionPreference = "Stop"
$action = $env:SRG_ACTION
$artifactPrefix = $env:SRG_ARTIFACT_NAME
if ([string]::IsNullOrWhiteSpace($artifactPrefix)) {
  $artifactPrefix = "srg-result"
}
$artifactPrefix = $artifactPrefix.Trim()
$artifactPrefix = $artifactPrefix -replace '[\\/:*?"<>|]', '-'
$artifactDir = Join-Path $PWD "artifacts"
$exePath = Join-Path $PWD "target\release\srg.exe"
$logPath = Join-Path $artifactDir ("{0}-console.log" -f $artifactPrefix)
$dataPath = Join-Path $PWD "random_data.txt"
$copiedDataPath = Join-Path $artifactDir ("{0}-random_data.txt" -f $artifactPrefix)
New-Item -ItemType Directory -Force $artifactDir | Out-Null
Remove-Item $dataPath -Force -ErrorAction SilentlyContinue
Set-Content -Path $logPath -Value "" -Encoding utf8NoBOM
"log_artifact_path=$logPath" >> $env:GITHUB_OUTPUT
function Require-NonEmpty([string] $value, [string] $name) {
  if ([string]::IsNullOrWhiteSpace($value)) {
    throw "$name is required for action '$action'."
  }
}
function Parse-UInt64([string] $value, [string] $name) {
  $parsed = [ulong]0
  if (-not [ulong]::TryParse($value, [ref]$parsed)) {
    throw "$name must be a valid unsigned integer."
  }
  return $parsed
}
function Parse-Int64([string] $value, [string] $name) {
  $parsed = [long]0
  if (-not [long]::TryParse($value, [ref]$parsed)) {
    throw "$name must be a valid integer."
  }
  return $parsed
}
function Parse-Int32([string] $value, [string] $name) {
  $parsed = [int]0
  if (-not [int]::TryParse($value, [ref]$parsed)) {
    throw "$name must be a valid integer."
  }
  return $parsed
}
function Parse-Double([string] $value, [string] $name) {
  $parsed = [double]0
  if (-not [double]::TryParse($value, [System.Globalization.NumberStyles]::Float -bor [System.Globalization.NumberStyles]::AllowThousands, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$parsed)) {
    throw "$name must be a valid floating-point number."
  }
  if (-not [double]::IsFinite($parsed)) {
    throw "$name must be finite."
  }
  return $parsed
}
function Start-SrgProcess {
  $proc = New-Object System.Diagnostics.Process
  $proc.StartInfo.FileName = $exePath
  $proc.StartInfo.WorkingDirectory = $PWD
  $proc.StartInfo.UseShellExecute = $false
  $proc.StartInfo.RedirectStandardInput = $true
  $proc.StartInfo.RedirectStandardOutput = $true
  $proc.StartInfo.RedirectStandardError = $true
  $proc.StartInfo.StandardOutputEncoding = [System.Text.Encoding]::UTF8
  $proc.StartInfo.StandardErrorEncoding = [System.Text.Encoding]::UTF8
  $null = $proc.Start()
  return $proc
}
function Finish-SrgProcess([System.Diagnostics.Process] $proc) {
  $stdout = $proc.StandardOutput.ReadToEnd()
  $stderr = $proc.StandardError.ReadToEnd()
  $proc.WaitForExit()
  $combined = if ([string]::IsNullOrEmpty($stderr)) {
    $stdout
  } else {
    "$stdout`r`n[stderr]`r`n$stderr"
  }
  Set-Content -Path $logPath -Value $combined -Encoding utf8NoBOM
  if ($proc.ExitCode -ne 0) {
    throw "srg.exe exited with code $($proc.ExitCode). See $logPath."
  }
}
function Run-SrgWithLines([string[]] $lines) {
  $proc = Start-SrgProcess
  foreach ($line in $lines) {
    $proc.StandardInput.WriteLine($line)
  }
  $proc.StandardInput.Close()
  Finish-SrgProcess $proc
}
try {
  switch ($action) {
    "generate-single" {
      Run-SrgWithLines @("6", "3", "q")
    }
    "generate-multiple" {
      $count = Parse-UInt64 $env:SRG_COUNT "count"
      if ($count -lt 1 -or $count -gt 100000) {
        throw "count must be between 1 and 100000."
      }
      Run-SrgWithLines @("6", "4", "$count", "q")
    }
    "ladder" {
      $players = $env:SRG_PLAYERS
      $results = $env:SRG_RESULTS
      Require-NonEmpty $players "players"
      Require-NonEmpty $results "results"
      Run-SrgWithLines @("6", "1", $players, $results, "q")
    }
    "random-integer" {
      $min = Parse-Int64 $env:SRG_INT_MIN "int_min"
      $max = Parse-Int64 $env:SRG_INT_MAX "int_max"
      if ($max -lt $min) {
        throw "int_max must be greater than or equal to int_min."
      }
      Run-SrgWithLines @("6", "2", "1", "$min", "$max", "q")
    }
    "random-float" {
      $min = Parse-Double $env:SRG_FLOAT_MIN "float_min"
      $max = Parse-Double $env:SRG_FLOAT_MAX "float_max"
      if ($max -lt $min) {
        throw "float_max must be greater than or equal to float_min."
      }
      Run-SrgWithLines @("6", "2", "2", "$min", "$max", "q")
    }
    "time-sync-observe" {
      $host = $env:SRG_TIME_HOST
      $observeSeconds = Parse-Int32 $env:SRG_OBSERVE_SECONDS "observe_seconds"
      Require-NonEmpty $host "time_host"
      if ($observeSeconds -lt 1 -or $observeSeconds -gt 60) {
        throw "observe_seconds must be between 1 and 60."
      }
      $proc = Start-SrgProcess
      $proc.StandardInput.WriteLine("6")
      $proc.StandardInput.WriteLine("5")
      $proc.StandardInput.WriteLine($host)
      $proc.StandardInput.WriteLine("")
      Start-Sleep -Seconds $observeSeconds
      $proc.StandardInput.WriteLine("")
      $proc.StandardInput.WriteLine("q")
      $proc.StandardInput.Close()
      Finish-SrgProcess $proc
    }
    default {
      throw "Unsupported action: $action"
    }
  }
  if (Test-Path $dataPath) {
    Copy-Item $dataPath $copiedDataPath -Force
    "data_artifact_path=$copiedDataPath" >> $env:GITHUB_OUTPUT
  }
} catch {
  Add-Content -Path $logPath -Value ("`r`n[workflow-error]`r`n" + $_.Exception.Message) -Encoding utf8NoBOM
  throw
}
