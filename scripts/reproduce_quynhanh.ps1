param(
    [string]$DataRoot = "../HKDN-261-20261016T025052Z-1-001/HKDN-261/data",
    [string]$Python = "python",
    [int[]]$Seeds = @(0, 1, 2)
)
$ErrorActionPreference = "Stop"
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    foreach ($signalSeed in $Seeds) {
        $signalInput = Join-Path $DataRoot "d1/synthetic/v8_az360/seed_$signalSeed"
        $signalCheckpoint = Join-Path $DataRoot "seed_$signalSeed/base.pt"
        & $Python scripts/export_signals.py --run_dir $signalInput --checkpoint $signalCheckpoint `
            --source cached --out_dir "runs/signals/cached/synthetic/v8_az360/seed_$signalSeed" `
            --figure_dir runs/signals/figures/cached
        if ($LASTEXITCODE -ne 0) { throw "Cached export failed for seed $signalSeed" }
        & $Python scripts/export_signals.py --run_dir $signalInput --checkpoint $signalCheckpoint `
            --source checkpoint --config results/phase0/config_used.yaml `
            --signals random opacity gradmag visibility gradient_inconsistency `
            --out_dir "runs/signals/recomputed/synthetic/v8_az360/seed_$signalSeed" `
            --figure_dir runs/signals/figures/recomputed
        if ($LASTEXITCODE -ne 0) { throw "Checkpoint export failed for seed $signalSeed" }
    }
    & $Python scripts/analyze_signal_results.py --data_root $DataRoot --seeds $Seeds
    if ($LASTEXITCODE -ne 0) { throw "Signal analysis failed" }
    # Keep the complete 81-figure experiment output in gitignored runs/.
    # Publish only eight representative seed-0 / Top-5% figures into the PR.
    if ($Seeds -contains 0) {
        $signalRepresentatives = @{
            cached = @("random", "opacity", "gradmag", "visibility")
            recomputed = @("opacity", "gradmag", "visibility", "gradient_inconsistency")
        }
        foreach ($signalSource in $signalRepresentatives.Keys) {
            $signalDestination = Join-Path "results/signals" $signalSource
            New-Item -ItemType Directory -Path $signalDestination -Force | Out-Null
            foreach ($signalName in $signalRepresentatives[$signalSource]) {
                $signalFigure = "synthetic_v8_az360_seed0_${signalName}_top5.png"
                $signalFigureSource = Join-Path "runs/signals/figures/$signalSource" $signalFigure
                Copy-Item -LiteralPath $signalFigureSource -Destination $signalDestination -Force
            }
        }
    }
} finally {
    Pop-Location
}
