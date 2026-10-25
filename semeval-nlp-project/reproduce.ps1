$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (-not (Test-Path '.venv/Scripts/python.exe')) {
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Failed to create virtual environment.' }
    }
    & ./.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & ./.venv/Scripts/python.exe reproduce.py
    if ($LASTEXITCODE -ne 0) { throw 'Reproduction failed. See reproduction/logs.' }
} finally {
    Pop-Location
}
