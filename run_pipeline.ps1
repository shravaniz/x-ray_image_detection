# run_pipeline.ps1 - one-command reproduction of the paper:
# "Ensemble deep-learning networks for automated osteoarthritis grading in knee X-ray images"
# (Pi et al., Sci Rep 13:22887, 2023, https://doi.org/10.1038/s41598-023-50210-4)
#
# Usage:
#   .\run_pipeline.ps1                     # train all 8 CNNs, TTA test, score, 6-model ensemble
#   .\run_pipeline.ps1 -All8               # ensemble with all 8 models instead of the paper's 6
#   .\run_pipeline.ps1 -SkipTrain          # use existing checkpoints (resume after training)
#   .\run_pipeline.ps1 -SkipTrain -SkipTest
#
# Paper targets: 6-model mix voting 76.93% acc / 0.7665 F1 | 8-model mix voting 76.33% / 0.7640

param(
    [switch]$All8,                       # ensemble with all 8 models (paper default: 6)
    [switch]$SkipTrain,                  # skip training for models that already have 5-fold checkpoints
    [switch]$SkipTest,                   # skip TTA inference entirely (use existing submission CSVs)
    [double]$Threshold = 0.65,           # score_auto printing threshold
    [string]$DataDir = '../KneeXrayData/KneeXrayData/ClsKLData/kneeKL224/train'  # class folders 0..4 with PNGs
)

$ErrorActionPreference = 'Continue'

# Paper's Table 3 optimized (best square) input sizes
$allModels = @(
    @{ Name = 'densenet_161';       Size = 512 },
    @{ Name = 'efficientnet_b5';    Size = 448 },
    @{ Name = 'efficientnet_v2_s';  Size = 456 },
    @{ Name = 'regnet_y_8gf';       Size = 384 },
    @{ Name = 'resnet_101';         Size = 384 },
    @{ Name = 'resnext_50_32x4d';   Size = 512 },
    @{ Name = 'wide_resnet_50_2';   Size = 456 },
    @{ Name = 'shufflenet_v2_x2_0'; Size = 512 }
)

# Paper's headline ensemble (Table 6/7): DenseNet-161 and Wide-ResNet-50-2 are excluded
$ensembleNames = @('efficientnet_b5', 'efficientnet_v2_s', 'regnet_y_8gf',
                   'resnet_101', 'resnext_50_32x4d', 'shufflenet_v2_x2_0')
if ($All8) { $ensembleNames = $allModels | ForEach-Object { $_.Name } }

Set-Location (Join-Path $PSScriptRoot 'OAI-KL')

function Banner([string]$msg) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
}

function FoldCheckpointsExist($m) {
    # training for a model counts as done only when all 5 folds have checkpoints
    # (merged main.py saves {N}fold_best.pt; older runs may have {N}fold_epoch*.pt)
    $dir = "./models/$($m.Name)/($($m.Size), $($m.Size))"
    if (-not (Test-Path $dir)) { return $false }
    foreach ($f in 1..5) {
        $count = (Get-ChildItem $dir -Filter "${f}fold_best.pt" -ErrorAction SilentlyContinue | Measure-Object).Count +
                 (Get-ChildItem $dir -Filter "${f}fold_epoch*.pt" -ErrorAction SilentlyContinue | Measure-Object).Count
        if ($count -eq 0) { return $false }
    }
    return $true
}

# ---------------------------------------------------------------- Stage 1: train
if (-not $SkipTrain) {
    Banner "STAGE 1/4 - Training 8 CNNs (5-fold CV each, ~30 epochs max)"
    foreach ($m in $allModels) {
        if (FoldCheckpointsExist $m) {
            Write-Host "SKIP  $($m.Name) @ $($m.Size) - all 5 folds already trained" -ForegroundColor Yellow
            continue
        }
        Banner "Train: $($m.Name) @ $($m.Size)x$($m.Size)"
        # --skip-test-evaluation: main.py's built-in evaluate.py uses a hardcoded data path
        # that does not match this machine; Stages 2-4 below handle test scoring instead.
        python main.py -m $m.Name -i $m.Size --data-dir $DataDir --class-weighting none --skip-test-evaluation
        if ($LASTEXITCODE -ne 0) { Write-Host "WARNING: training exited with code $LASTEXITCODE" -ForegroundColor Red }
    }
}
else { Write-Host "Stage 1 skipped (-SkipTrain)" -ForegroundColor Yellow }

# ---------------------------------------------------------------- Stage 2: TTA inference
Banner "STAGE 2/4 - TTA test predictions (one CSV per checkpoint)"
foreach ($m in $allModels) {
    $dir = "./models/$($m.Name)/($($m.Size), $($m.Size))"
    if ((Get-ChildItem $dir -Filter *.pt -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0) {
        Write-Host "SKIP  $($m.Name) - no checkpoints in $dir" -ForegroundColor Yellow
        continue
    }
    if ($SkipTest) {
        Write-Host "SKIP  $($m.Name) - -SkipTest" -ForegroundColor Yellow
        continue
    }
    python test_auto.py -m $m.Name -i $m.Size   # idempotent: skips CSVs already generated
    if ($LASTEXITCODE -ne 0) { Write-Host "WARNING: test_auto exited with code $LASTEXITCODE" -ForegroundColor Red }
}

# ---------------------------------------------------------------- Stage 3: per-model scores
Banner "STAGE 3/4 - Per-model accuracy / F1 scores"
foreach ($m in $allModels) {
    $dir = "./submission/$($m.Name)/($($m.Size), $($m.Size))"
    if ((Get-ChildItem $dir -Filter *.csv -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0) {
        Write-Host "SKIP  $($m.Name) - no submission CSVs" -ForegroundColor Yellow
        continue
    }
    python score_auto.py -m $m.Name -i $m.Size -t $Threshold
}

# ---------------------------------------------------------------- Stage 4: ensemble
Banner "STAGE 4/4 - Ensemble ($($ensembleNames.Count) models, hard/soft/mix voting)"
Write-Host "Ensemble members: $($ensembleNames -join ', ')"

# test_ensemble.py reads EVERY csv in the flat ./submission/ folder - clear leftovers first
$leftover = (Get-ChildItem ./submission -Filter *.csv -ErrorAction SilentlyContinue | Measure-Object).Count
if ($leftover -gt 0) {
    Write-Host "Clearing $leftover old csv(s) from ./submission/" -ForegroundColor Yellow
    Get-ChildItem ./submission -Filter *.csv | Remove-Item -Force
}

foreach ($name in $ensembleNames) {
    $m = $allModels | Where-Object { $_.Name -eq $name }
    $src = "./submission/$($m.Name)/($($m.Size), $($m.Size))"
    $n = (Get-ChildItem $src -Filter *_submission.csv -ErrorAction SilentlyContinue | Measure-Object).Count
    if ($n -gt 0) {
        Get-ChildItem $src -Filter *_submission.csv | Copy-Item -Destination ./submission
    }
    Write-Host "Added $n csv(s) from $($m.Name)"
}

python test_ensemble.py
if ($LASTEXITCODE -ne 0) { Write-Host "WARNING: test_ensemble exited with code $LASTEXITCODE" -ForegroundColor Red }

# Score the three ensemble outputs (10fold_epoch10 = hard, 11 = soft, 12 = mix)
$scorePy = @'
import pandas as pd, glob
from sklearn.metrics import accuracy_score, f1_score
t = pd.read_csv("./KneeXray/test/test_correct.csv", names=["d", "l"], skiprows=1)["l"].tolist()
for f in sorted(glob.glob("./submission/10fold_epoch*.csv")):
    s = pd.read_csv(f, names=["d", "l", "pc", "pp", "p0", "p1", "p2", "p3", "p4"], skiprows=1)["l"].tolist()
    mode = {"10": "hard", "11": "soft", "12": "mix"}[f.split("epoch")[1][:2]]
    print("ENSEMBLE %-4s : acc = %.4f  f1_macro = %.4f  f1_weighted = %.4f" % (
        mode, accuracy_score(t, s), f1_score(t, s, average="macro"), f1_score(t, s, average="weighted")))
'@
$scorePy | Out-File -FilePath "$env:TEMP\score_ensemble.py" -Encoding utf8
python "$env:TEMP\score_ensemble.py"

Banner "PIPELINE COMPLETE"
Write-Host "Paper reference: 6-model mix voting 76.93% acc / 0.7665 F1 | 8-model mix 76.33% / 0.7640"
Write-Host "Per-model scores were printed in Stage 3; ensemble scores above."
