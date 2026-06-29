<#
.SYNOPSIS
Find and optionally download RunPod H5 artifacts without scanning the whole bucket.

.DESCRIPTION
This checks exact likely keys used by the Pod notebooks and repo mirror. Use it
when a broad aws s3 sync or recursive aws s3 ls stays quiet for a long time.
#>

[CmdletBinding()]
param(
    [string]$Source = "s3://0wov6gbp6j/",
    [string]$Destination = "C:\runpod_data\",
    [string]$Region = "us-ks-2",
    [string]$EndpointUrl = "https://s3api-us-ks-2.runpod.io",
    [string]$AwsCliPath = "aws",
    [switch]$Download
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command $AwsCliPath -ErrorAction SilentlyContinue)) {
    throw "AWS CLI not found: $AwsCliPath"
}

$sourceRoot = $Source.TrimEnd("/") + "/"
$candidateKeys = @(
    "stable-query-latent/game_review_data/embedding_h5.h5",
    "stable-query-latent/game_review_data/embedding_h5.h5.cloud_manifest.json",
    "stable-query-latent/game_review_data/embedding_h5.h5.incloud_manifest.json",
    "stable-query-latent/game_review_data/build_new_gamedata/text_h5.h5",
    "stable-query-latent/game_review_data/build_new_gamedata/text_h5.h5.manifest.json",

    "workspace/stable-query-latent/game_review_data/embedding_h5.h5",
    "workspace/stable-query-latent/game_review_data/embedding_h5.h5.cloud_manifest.json",
    "workspace/stable-query-latent/game_review_data/embedding_h5.h5.incloud_manifest.json",
    "workspace/stable-query-latent/game_review_data/build_new_gamedata/text_h5.h5",
    "workspace/stable-query-latent/game_review_data/build_new_gamedata/text_h5.h5.manifest.json",

    "game_review_data/embedding_h5.h5",
    "game_review_data/embedding_h5.h5.cloud_manifest.json",
    "game_review_data/embedding_h5.h5.incloud_manifest.json",
    "game_review_data/build_new_gamedata/text_h5.h5",
    "game_review_data/build_new_gamedata/text_h5.h5.manifest.json"
)

$matches = New-Object System.Collections.Generic.List[string]

foreach ($key in $candidateKeys) {
    $uri = $sourceRoot + $key
    Write-Host "Checking $key"
    $output = & $AwsCliPath s3 ls `
        --region $Region `
        --endpoint-url $EndpointUrl `
        $uri 2>$null

    if ($LASTEXITCODE -eq 0 -and $output) {
        $matches.Add($key)
        Write-Host "  FOUND $key"
    }
}

$uniqueMatches = $matches | Sort-Object -Unique

if (-not $uniqueMatches) {
    Write-Host ""
    Write-Host "No text_h5.h5 or embedding_h5.h5 files were found at the expected exact keys."
    Write-Host "That usually means the H5 files were not uploaded to this S3 bucket, or they are under an unexpected prefix."
    exit 2
}

Write-Host ""
Write-Host "Matched files:"
$uniqueMatches | ForEach-Object { Write-Host "  $_" }

if (-not $Download) {
    Write-Host ""
    Write-Host "Add -Download to copy these files into $Destination"
    exit 0
}

foreach ($key in $uniqueMatches) {
    $src = $sourceRoot + $key
    $dst = Join-Path $Destination $key
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
    Write-Host "Downloading $key"
    & $AwsCliPath s3 cp `
        --region $Region `
        --endpoint-url $EndpointUrl `
        $src `
        $dst
}
