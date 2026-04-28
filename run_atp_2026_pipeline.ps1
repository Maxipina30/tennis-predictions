param(
    [string]$Python = "python",
    [int]$Year = 2026,
    [string]$Out = "files/processed/atp_2026",
    [double]$Delay = 1.0,
    [switch]$CompletedOnly,
    [int]$MaxTournaments = 0,
    [int]$MaxDetailsPerTournament = 0
)

$ErrorActionPreference = "Stop"

& $Python -m pip install -r requirements.txt

$scrapeArgs = @(
    "src/01_scrape_tennisexplorer.py",
    "--batch-atp",
    "--year", "$Year",
    "--out", $Out,
    "--delay", "$Delay",
    "--levels", "grand_slam", "masters_1000", "atp_500", "atp_250"
)

if ($CompletedOnly) {
    $scrapeArgs += "--completed-only"
}

if ($MaxTournaments -gt 0) {
    $scrapeArgs += @("--max-tournaments", "$MaxTournaments")
}

if ($MaxDetailsPerTournament -gt 0) {
    $scrapeArgs += @("--max-details-per-tournament", "$MaxDetailsPerTournament")
}

& $Python @scrapeArgs
& $Python "src/02_build_features.py" --matches "$Out/matches.csv" --out "$Out/features_no_leakage.csv"
