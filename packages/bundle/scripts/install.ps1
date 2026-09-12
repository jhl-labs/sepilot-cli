# install.ps1 - download, verify, and install the standalone `sepilot` binary on Windows.
#
# One-liner:
#   irm https://raw.githubusercontent.com/jhl-labs/sepilot-cli/main/packages/bundle/scripts/install.ps1 | iex
#
# Self-contained: assumes no repo checkout. Re-running upgrades in place.
#
# Environment overrides:
#   $env:SEPILOT_INSTALL_DIR  install directory (default: "$env:LOCALAPPDATA\Programs\sepilot")
#   $env:SEPILOT_VERSION      "latest" (default) or a version like "0.2.10" / "v0.2.10"
#   $env:SEPILOT_REPO         GitHub "owner/repo" (default: jhl-labs/sepilot-cli)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$repo = if ($env:SEPILOT_REPO) { $env:SEPILOT_REPO } else { 'jhl-labs/sepilot-cli' }
$version = if ($env:SEPILOT_VERSION) { $env:SEPILOT_VERSION } else { 'latest' }
$installDir = if ($env:SEPILOT_INSTALL_DIR) { $env:SEPILOT_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA 'Programs\sepilot' }

# -- Detect architecture -----------------------------------------------------
try {
  $archEnum = [System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString()
} catch {
  $archEnum = $env:PROCESSOR_ARCHITECTURE
}
switch -Wildcard ($archEnum) {
  'X64'   { $arch = 'x64' }
  'AMD64' { $arch = 'x64' }
  'Arm64' { $arch = 'arm64' }
  'ARM64' { $arch = 'arm64' }
  default { throw "install.ps1: unsupported architecture '$archEnum' - only x64 is currently shipped for Windows." }
}
if ($arch -ne 'x64') {
  throw "install.ps1: no Windows '$arch' build is published yet - only sepilot-windows-x64.exe is available."
}

$asset = "sepilot-windows-$arch.exe"

# -- Resolve download URLs ---------------------------------------------------
if ($version -eq 'latest') {
  $baseUrl = "https://github.com/$repo/releases/latest/download"
  $versionLabel = 'latest'
} else {
  $v = $version -replace '^v', ''
  $baseUrl = "https://github.com/$repo/releases/download/v$v"
  $versionLabel = "v$v"
}
$binUrl = "$baseUrl/$asset"
$shaUrl = "$baseUrl/$asset.sha256"

# -- Download to a temp dir --------------------------------------------------
$tmpDir = Join-Path ([System.IO.Path]::GetTempPath()) ("sepilot-install-" + [System.Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
$cleanup = { Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $tmpDir }
try {
  $tmpBin = Join-Path $tmpDir $asset
  $tmpSha = Join-Path $tmpDir "$asset.sha256"

  Write-Host "install.ps1: downloading $asset ($versionLabel) for windows-$arch..."
  try {
    Invoke-WebRequest -Uri $binUrl -OutFile $tmpBin -UseBasicParsing
  } catch {
    throw "install.ps1: failed to download $binUrl - check the version/repo, or that the release has a $asset asset. ($_)"
  }
  try {
    Invoke-WebRequest -Uri $shaUrl -OutFile $tmpSha -UseBasicParsing
  } catch {
    throw "install.ps1: failed to download $shaUrl - the release is missing the checksum file. ($_)"
  }

  # -- Verify sha256 ---------------------------------------------------------
  # The .sha256 file is "<hex>  <filename>" (sha256sum format) or a bare "<hex>";
  # take the first whitespace-delimited token either way.
  $shaText = (Get-Content -Raw -LiteralPath $tmpSha)
  $expected = ($shaText.Trim() -split '\s+', 2)[0].ToLowerInvariant()
  if ([string]::IsNullOrWhiteSpace($expected)) {
    throw "install.ps1: could not read an expected SHA-256 from $asset.sha256."
  }
  $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $tmpBin).Hash.ToLowerInvariant()
  if ($expected -ne $actual) {
    throw "install.ps1: checksum mismatch for $asset`n  expected: $expected`n  actual:   $actual"
  }
  Write-Host "install.ps1: checksum OK ($actual)"

  # -- Install ---------------------------------------------------------------
  New-Item -ItemType Directory -Force -Path $installDir | Out-Null
  $dest = Join-Path $installDir 'sepilot.exe'
  try {
    Move-Item -Force -LiteralPath $tmpBin -Destination $dest
  } catch {
    # Can't overwrite a running sepilot.exe (locked / in use). Drop a .new and
    # tell the user how to finish - mirrors `sepilot upgrade`'s fallback.
    $destNew = "$dest.new"
    Remove-Item -Force -ErrorAction SilentlyContinue $destNew
    Move-Item -Force -LiteralPath $tmpBin -Destination $destNew
    throw "install.ps1: could not replace '$dest' (it may be in use). The new build is at '$destNew'. Close all sepilot processes, then run:`n  Move-Item -Force '$destNew' '$dest'`n(or just re-run this installer)."
  }

  # -- Add install dir to the user PATH if absent ----------------------------
  $userPath = [Environment]::GetEnvironmentVariable('PATH', 'User')
  if ($null -eq $userPath) { $userPath = '' }
  $target = $installDir.TrimEnd('\')
  $alreadyOnPath = @($userPath -split ';' | Where-Object { $_.TrimEnd('\') -ieq $target }).Length -gt 0
  if (-not $alreadyOnPath) {
    $newUserPath = if ([string]::IsNullOrEmpty($userPath)) { $installDir } else { "$userPath;$installDir" }
    [Environment]::SetEnvironmentVariable('PATH', $newUserPath, 'User')
    Write-Host "install.ps1: added $installDir to your user PATH. Open a new terminal for the PATH change to take effect."
  }

  Write-Host ""
  Write-Host "sepilot installed to $dest. Run ``sepilot --help`` to get started."
} finally {
  & $cleanup
}
