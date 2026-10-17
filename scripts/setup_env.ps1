# Recreate the reviewed offline environment: Python 3.11.9 plus the full lock.
#
# Usage, from the repository root:
#   powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1 -RunTests
#
# The runtime fingerprint in revision224.py compares platform.python_version()
# exactly, so a different patch release (for example 3.11.10) is refused rather
# than silently accepted. No API key is read and no model call is made.
param(
    [switch]$RunTests,
    [switch]$Recreate
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Expected = (Get-Content (Join-Path $Root '.python-version') -Raw).Trim()
$Venv = Join-Path $Root '.venv'
$Python = Join-Path $Venv 'Scripts\python.exe'

$Found = (& py -3.11 -c "import platform; print(platform.python_version())" 2>$null)
if ($Found -ne $Expected) {
    throw "Python $Expected is required (found '$Found' via 'py -3.11'). Install it from https://www.python.org/downloads/release/python-3119/"
}

if ($Recreate -and (Test-Path $Venv)) {
    Remove-Item -Recurse -Force $Venv
}
if (-not (Test-Path $Python)) {
    & py -3.11 -m venv $Venv
}
$VenvVersion = (& $Python -c "import platform; print(platform.python_version())")
if ($VenvVersion -ne $Expected) {
    throw "Existing .venv uses Python $VenvVersion, not $Expected. Rerun with -Recreate."
}

& $Python -m pip install --quiet --disable-pip-version-check -r (Join-Path $Root 'requirements.lock.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $Python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Installed dependencies are inconsistent.' }
& $Python -c "import platform, numpy, pandas, networkx, openai, matplotlib, scipy; print('Python', platform.python_version(), '| numpy', numpy.__version__, '| pandas', pandas.__version__, '| networkx', networkx.__version__, '| openai', openai.__version__, '| matplotlib', matplotlib.__version__, '| scipy', scipy.__version__)"

if ($RunTests) {
    # Keep any ambient credential out of the offline suite.
    Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
    Push-Location $Root
    try {
        & $Python -B -m unittest discover -s tests
        if ($LASTEXITCODE -ne 0) { throw 'Offline test suite failed.' }
    } finally {
        Pop-Location
    }
}
