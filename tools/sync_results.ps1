<#
.SYNOPSIS
Pull ONLY the light-weight eval/test result files from the RunPod bucket.

.DESCRIPTION
The heavy sibling (sync_runpod_artifacts.ps1) also downloads checkpoints and
H5s (tens of GB). This one grabs just the analysis artifacts:
  - grid_metrics.json / champions.json        (battery results + selection)
  - per-combo eval_report.json                (full downstream battery detail)
  - per-combo dual_probe_history.tsv          (in-training convergence curves)
  - sweep summaries / report / manifest, final_best_eval, raw_test_data CSVs
  - archived combo_status.json / collection_manifest.json / paper_handoff.md

Usage:
  .\tools\sync_results.ps1            # download into C:\runpod_data\
  .\tools\sync_results.ps1 -DryRun    # preview what would be copied
#>

[CmdletBinding()]
param(
    [string]$Source = "s3://0wov6gbp6j/",
    [string]$Destination = "C:\runpod_data\",
    [string]$Region = "us-ks-2",
    [string]$EndpointUrl = "https://s3api-us-ks-2.runpod.io",
    [string]$AwsCliPath = "aws",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$includes = @(
    "*cloud_full_sweep_a100/grid_metrics.json",
    "*cloud_full_sweep_a100/champions.json",
    "*cloud_full_sweep_a100/combo_timings.json",
    "*cloud_full_sweep_a100/realtext_grid_metrics.json",
    "*cloud_full_sweep_a100/champions_namerank.json",
    "*cloud_full_sweep_a100/*/real_text_grid.json",
    "*cloud_full_sweep_a100/*/eval_report.json",
    "*cloud_full_sweep_a100/*/dual_probe_history.tsv",
    "*cloud_full_sweep_a100/*/dual_probe_history.jsonl",
    "*cloud_full_sweep_a100/data_view_sweep_summary.csv",
    "*cloud_full_sweep_a100/data_view_sweep_summary.json",
    "*cloud_full_sweep_a100/DATA_VIEW_SWEEP_REPORT.md",
    "*cloud_full_sweep_a100/sweep_manifest.json",
    "*cloud_full_sweep_a100/final_best_eval/*",
    "*cloud_full_sweep_a100/raw_test_data/*",
    "*stable_query_latent_artifacts/*/combo_status.json",
    "*stable_query_latent_artifacts/*/collection_manifest.json",
    "*stable_query_latent_artifacts/*/paper_handoff.md",
    "*stable_query_latent_artifacts/*/git_info.json"
)

$cliArgs = @("s3", "cp", $Source, $Destination, "--recursive",
             "--region", $Region, "--endpoint-url", $EndpointUrl,
             "--exclude", "*")
foreach ($pattern in $includes) {
    $cliArgs += @("--include", $pattern)
}
if ($DryRun) {
    $cliArgs += "--dryrun"
}

Write-Host "RunPod results sync (light: no checkpoints, no H5s)"
Write-Host "  Source      : $Source"
Write-Host "  Destination : $Destination"
Write-Host "  Mode        : $(if ($DryRun) { 'dry-run preview' } else { 'download' })"
Write-Host ""

& $AwsCliPath @cliArgs
exit $LASTEXITCODE
