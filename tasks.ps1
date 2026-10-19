<#
.SYNOPSIS
  PortfolioIQ task runner (Windows PowerShell replacement for make).
.EXAMPLE
  .\tasks.ps1 setup | test | lint | sample | full | airflow-up | airflow-down | mlflow-ui | secrets
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet("setup", "test", "lint", "sample", "full", "airflow-up", "airflow-down", "airflow-validate", "mlflow-ui", "secrets", "help")]
    [string]$Target = "help"
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Venv = Join-Path $Root ".venv"
$Py = Join-Path $Venv "Scripts\python.exe"
$Logs = Join-Path $Root "logs"
New-Item -ItemType Directory -Force $Logs | Out-Null
$env:PYTHONPATH = Join-Path $Root "src"
$env:PYTHONUTF8 = "1"

function Invoke-Py([string[]]$PyArgs) {
    & $Py @PyArgs
    if ($LASTEXITCODE -ne 0) { throw "python $($PyArgs -join ' ') failed with exit code $LASTEXITCODE" }
}

switch ($Target) {
    "setup" {
        if (-not (Test-Path $Py)) { py -3.11 -m venv $Venv }
        Invoke-Py @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
        Invoke-Py @("-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu")
        Invoke-Py @("-m", "pip", "install", "-e", "$Root[dev]")
        Write-Host "Setup OK. Activate with .venv\Scripts\Activate.ps1"
    }
    "test" {
        Invoke-Py @("-m", "pytest", (Join-Path $Root "tests"))
        Invoke-Py @("-m", "ruff", "check", $Root)
    }
    "lint" { Invoke-Py @("-m", "ruff", "check", $Root) }
    "sample" {
        $log = Join-Path $Logs ("sample_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
        Write-Host "Running sample pipeline, log: $log"
        & $Py -m portfolioiq.cli run-all --config (Join-Path $Root "configs\sample.yaml") 2>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) { throw "sample pipeline failed (see $log)" }
    }
    "full" {
        $log = Join-Path $Logs ("full_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
        Write-Host "Running full pipeline, log: $log"
        & $Py -m portfolioiq.cli run-all --config (Join-Path $Root "configs\default.yaml") 2>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) { throw "full pipeline failed (see $log)" }
    }
    "airflow-up" { docker compose -f (Join-Path $Root "docker-compose.yml") up -d }
    "airflow-down" { docker compose -f (Join-Path $Root "docker-compose.yml") down }
    "airflow-validate" {
        docker compose -f (Join-Path $Root "docker-compose.yml") --profile cli run --rm airflow-cli
    }
    "mlflow-ui" { Invoke-Py @("-m", "mlflow", "ui", "--backend-store-uri", ("file:///" + (Join-Path $Root "mlruns").Replace('\', '/')), "--port", "5000") }
    "secrets" { Invoke-Py @("-m", "detect_secrets", "scan", "--exclude-files", "(\.venv|data|mlruns|logs|\.git|dbt[/\\]target)") }
    default { Get-Help $PSCommandPath -Detailed }
}
