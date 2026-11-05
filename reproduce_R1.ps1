$ErrorActionPreference = 'Stop'
# Reproduce the revised manuscript (R1) from the prepared HBTA arrays.
# The submitted pipeline is run unchanged; the R1 steps only add new outputs.
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$sourceRoot = Join-Path $projectRoot 'src'
$studyRoot = Join-Path $projectRoot 'studies\topology_sensor_placement_hbta'
$paperBuild = Join-Path $projectRoot 'paper\outage_conditioned_prioritization\build_topology_coverage_paper.ps1'
$revisedPdf = Join-Path $projectRoot 'output\pdf\topology_coverage_restoration_R1.pdf'
$hasPaper = Test-Path -LiteralPath $paperBuild

python -c "import matplotlib,numpy,scipy,sklearn" | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Python dependencies are incomplete. Install requirements.txt.' }
if ($hasPaper -and -not (Get-Command pdflatex -ErrorAction SilentlyContinue)) { throw 'pdflatex is not available.' }
$prepared = Join-Path $projectRoot 'data\derived\hbta\edge_spectra_multicondition_10s.npz'
$teacher = Join-Path $projectRoot 'data\derived\hbta\hbta_teacher_edge_state.npz'
if (-not (Test-Path -LiteralPath $prepared) -or -not (Test-Path -LiteralPath $teacher)) {
    throw 'Prepared HBTA arrays are missing. Run git lfs pull.'
}

# Optional: verify the raw benchmark file if config_local.json points to it.
$localConfig = Join-Path $projectRoot 'config_local.json'
if (Test-Path -LiteralPath $localConfig) {
    $configuration = Get-Content -Raw -LiteralPath $localConfig | ConvertFrom-Json
    $rawHdf5 = [System.IO.Path]::Combine([string]$configuration.raw_data_dir, 'data_100Hz.h5')
    if (Test-Path -LiteralPath $rawHdf5 -ErrorAction SilentlyContinue) {
        $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $rawHdf5).Hash.ToLowerInvariant()
        if ($hash -ne '467e9ecd87b367a225290f58e382657548ca81b235b08c70b7f1661174219650') {
            # The R1 pipeline uses only the prepared arrays, whose hashes are checked by
            # verify_R1_results.py; a different raw-file version is reported, not fatal.
            Write-Warning "Raw HBTA file differs from the version used to prepare the arrays (SHA-256 $hash)."
        } else {
            Write-Host 'Raw HBTA file found and checksum verified.'
        }
    } else {
        Write-Host 'Raw HBTA file not found; continuing from the prepared arrays.'
    }
}

$env:PYTHONPATH = $sourceRoot
Push-Location $studyRoot
try {
    # Submitted pipeline (unchanged scripts)
    python src\run_virtual_damage_placement.py --config configs\virtual_damage_pareto.json --sensitivity-only
    if ($LASTEXITCODE -ne 0) { throw 'q_md generation failed.' }
    python src\run_topology_coverage_restoration.py --output outputs\topology_coverage_restoration_10sensor --selection-only
    if ($LASTEXITCODE -ne 0) { throw 'Restoration-priority generation failed.' }
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

    # R1 additions (new scripts, new output folder)
    python src\make_r1_link_level_all_states.py
    if ($LASTEXITCODE -ne 0) { throw 'R1 link-level analysis failed.' }
    python src\make_r1_fig7_all_states.py
    if ($LASTEXITCODE -ne 0) { throw 'R1 Figure 7 generation failed.' }
    python src\make_r1_spring_model_figure.py
    if ($LASTEXITCODE -ne 0) { throw 'R1 spring-model figure generation failed.' }
}
finally {
    Pop-Location
}

python (Join-Path $projectRoot 'scripts\generate_tables.py')
if ($LASTEXITCODE -ne 0) { throw 'LaTeX table generation failed.' }
python (Join-Path $projectRoot 'scripts\verify_R1_results.py')
if ($LASTEXITCODE -ne 0) { throw 'R1 scientific-result verification failed.' }

if (-not $hasPaper) {
    Write-Host 'R1 results reproduced and verified. The manuscript sources are not part of this repository; PDF build skipped.'
    return
}
& $paperBuild
if ($LASTEXITCODE -ne 0) { throw 'Revised LaTeX build failed.' }
python (Join-Path $projectRoot 'scripts\verify_R1_results.py') --pdf $revisedPdf
if ($LASTEXITCODE -ne 0) { throw 'Revised PDF verification failed.' }

Write-Host "R1 reproduction complete: $revisedPdf"
