$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$localConfig = Join-Path $projectRoot 'config_local.json'
$sourceRoot = Join-Path $projectRoot 'src'
$studyRoot = Join-Path $projectRoot 'studies\topology_sensor_placement_hbta'
$paperBuild = Join-Path $projectRoot 'paper\original_submitted\build_topology_coverage_paper.ps1'
$hasPaper = Test-Path -LiteralPath $paperBuild
$finalPdf = Join-Path $projectRoot 'output\pdf\topology_coverage_restoration.pdf'

if (-not (Test-Path -LiteralPath $localConfig)) {
    throw 'Missing config_local.json. Copy config_local.example.json and set raw_data_dir.'
}
$configuration = Get-Content -Raw -LiteralPath $localConfig | ConvertFrom-Json
$rawDataDirectory = [System.IO.Path]::GetFullPath([string]$configuration.raw_data_dir)
$rawHdf5 = Join-Path $rawDataDirectory 'data_100Hz.h5'
$bridgePhoto = Join-Path $rawDataDirectory 'The Hell Bridge Test Arena.jpg'
if (-not (Test-Path -LiteralPath $rawHdf5)) { throw "Missing external HBTA raw data: $rawHdf5" }
if (-not (Test-Path -LiteralPath $bridgePhoto)) { throw "Missing bridge photo: $bridgePhoto" }
$expectedRawSha256 = '467e9ecd87b367a225290f58e382657548ca81b235b08c70b7f1661174219650'
$actualRawSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $rawHdf5).Hash.ToLowerInvariant()
if ($actualRawSha256 -ne $expectedRawSha256) { throw 'HBTA raw HDF5 checksum mismatch.' }

python -c "import matplotlib,numpy,scipy,sklearn,torch" | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Python dependencies are incomplete. Install requirements.txt.' }
if ($hasPaper -and -not (Get-Command pdflatex -ErrorAction SilentlyContinue)) { throw 'pdflatex is not available.' }

$prepared = Join-Path $projectRoot 'data\derived\hbta\edge_spectra_multicondition_10s.npz'
$teacher = Join-Path $projectRoot 'data\derived\hbta\hbta_teacher_edge_state.npz'
if (-not (Test-Path -LiteralPath $prepared) -or -not (Test-Path -LiteralPath $teacher)) {
    throw 'Prepared HBTA arrays are missing. Run git lfs pull.'
}

$env:PYTHONPATH = $sourceRoot
Push-Location $studyRoot
try {
    python src\run_virtual_damage_placement.py --config configs\virtual_damage_pareto.json --sensitivity-only
    if ($LASTEXITCODE -ne 0) { throw 'q_md generation failed.' }

    python src\run_topology_coverage_restoration.py --output outputs\topology_coverage_restoration_10sensor --selection-only
    if ($LASTEXITCODE -ne 0) { throw 'Restoration-priority generation failed.' }

    $contextFigures = Join-Path $studyRoot 'outputs\topology_coverage_restoration\figures'
    New-Item -ItemType Directory -Force -Path $contextFigures | Out-Null
    Copy-Item -LiteralPath $bridgePhoto -Destination (Join-Path $contextFigures 'fig_hbta_bridge_photo.jpg') -Force

    python src\make_hbta_system_context_figure.py `
        --source-npz ..\..\data\derived\hbta\edge_spectra_multicondition_10s.npz `
        --repository ..\..\src `
        --output-stem outputs\topology_coverage_restoration\figures\fig_hbta_sensor_topology_wide
    if ($LASTEXITCODE -ne 0) { throw 'HBTA system-context figure generation failed.' }

    python src\make_topology_coverage_concept_figure.py
    if ($LASTEXITCODE -ne 0) { throw 'Topology-coverage concept figure generation failed.' }
    python src\make_topology_coverage_restoration_figures.py
    if ($LASTEXITCODE -ne 0) { throw 'Restoration figure generation failed.' }
    python src\make_physical_consistency_analysis.py
    if ($LASTEXITCODE -ne 0) { throw 'DS1 physical-consistency analysis failed.' }
    python src\make_physical_consistency_figure.py
    if ($LASTEXITCODE -ne 0) { throw 'DS1 physical-consistency figure generation failed.' }
}
finally {
    Pop-Location
}

python (Join-Path $projectRoot 'scripts\generate_tables.py')
if ($LASTEXITCODE -ne 0) { throw 'LaTeX table generation failed.' }
python (Join-Path $projectRoot 'scripts\verify_final_results.py')
if ($LASTEXITCODE -ne 0) { throw 'Scientific-result verification failed.' }

if (-not $hasPaper) {
    Write-Host 'Results reproduced and verified. The manuscript sources are not part of this repository; PDF build skipped.'
    return
}
& $paperBuild
if ($LASTEXITCODE -ne 0) { throw 'Final LaTeX build failed.' }
python (Join-Path $projectRoot 'scripts\verify_final_results.py') --pdf $finalPdf
if ($LASTEXITCODE -ne 0) { throw 'Final PDF verification failed.' }

Write-Host "Reproduction complete: $finalPdf"
