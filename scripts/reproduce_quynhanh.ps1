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
            --figure_dir results/signals/cached
        if ($LASTEXITCODE -ne 0) { throw "Cached export failed for seed $signalSeed" }
        & $Python scripts/export_signals.py --run_dir $signalInput --checkpoint $signalCheckpoint `
            --source checkpoint --config results/phase0/config_used.yaml `
            --signals random opacity gradmag visibility gradient_inconsistency `
            --out_dir "runs/signals/recomputed/synthetic/v8_az360/seed_$signalSeed" `
            --figure_dir results/signals/recomputed
        if ($LASTEXITCODE -ne 0) { throw "Checkpoint export failed for seed $signalSeed" }
    }
    & $Python scripts/analyze_signal_results.py --data_root $DataRoot --seeds $Seeds
    if ($LASTEXITCODE -ne 0) { throw "Signal analysis failed" }
} finally {
    Pop-Location
}
