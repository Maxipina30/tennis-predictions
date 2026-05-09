param(
    [Parameter(Mandatory = $true)]
    [string]$Tournament,

    [Parameter(Mandatory = $true)]
    [string]$Round,

    [string]$Year = "2026",
    [string]$Label,
    [string]$Url,
    [string]$ReferenceDate = (Get-Date -Format "yyyy-MM-dd"),
    [int]$RawTodayOffsetDays = 0,
    [int]$RawTomorrowOffsetDays = 1,
    [double]$Delay = 0.5,
    [switch]$SkipTournamentScrape,
    [switch]$SkipRankingRefresh,
    [switch]$LaunchDashboard,
    [string]$Python = "venv312\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

$prepareArgs = @(
    "src/13_prepare_dashboard_target.py",
    "--tournament", $Tournament,
    "--round", $Round,
    "--year", $Year,
    "--reference-date", $ReferenceDate,
    "--raw-today-offset-days", "$RawTodayOffsetDays",
    "--raw-tomorrow-offset-days", "$RawTomorrowOffsetDays",
    "--delay", "$Delay"
)

if ($Label) { $prepareArgs += @("--label", $Label) }
if ($Url) { $prepareArgs += @("--url", $Url) }
if ($SkipTournamentScrape) { $prepareArgs += "--skip-tournament-scrape" }
if ($SkipRankingRefresh) { $prepareArgs += "--skip-ranking-refresh" }

& $Python @prepareArgs

if ($LaunchDashboard) {
    & $Python "-m" "streamlit" "run" "app.py" "--server.headless" "true" "--server.port" "8501"
}
