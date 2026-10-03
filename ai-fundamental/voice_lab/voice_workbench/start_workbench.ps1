$ErrorActionPreference = 'Stop'
$workbenchRoot = $PSScriptRoot
$pythonExecutable = Join-Path $workbenchRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable)) {
    Write-Host 'The local model environment is missing. Follow README.md preparation first.'
    exit 1
}
Write-Host 'Voice workbench: http://127.0.0.1:8872'
Write-Host 'Keep this window open. Press Ctrl+C to stop the local service.'
Push-Location -LiteralPath $workbenchRoot
try {
    & $pythonExecutable -X utf8 app.py
} finally {
    Pop-Location
}
