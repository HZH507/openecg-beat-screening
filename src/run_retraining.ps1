$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$projectPython = 'D:\openecg\.venv\Scripts\python.exe'
Set-Location -LiteralPath 'D:\openecg（1）'
$logPath = 'results\retraining_20261029.log'
Start-Transcript -LiteralPath $logPath -NoClobber
try {
    foreach ($weight in @(3,5,9)) {
        & $projectPython -u 'src\train_binary.py' --abnormal-weight $weight
        if ($LASTEXITCODE -ne 0) { throw "Binary weight $weight failed" }
    }
    & $projectPython -u 'src\train_classifier.py' --model cnn --arch light --sampler --lr 1e-3
    if ($LASTEXITCODE -ne 0) { throw 'Four-class CNN training failed' }
    & $projectPython -u 'src\train_classifier.py' --model mlp --arch light --lr 5e-4
    if ($LASTEXITCODE -ne 0) { throw 'MLP baseline training failed' }
    & $projectPython -u 'src\train_denoiser.py'
    if ($LASTEXITCODE -ne 0) { throw 'DnCNN training failed' }
    & $projectPython -u 'src\sweep_threshold.py'
    if ($LASTEXITCODE -ne 0) { throw 'Validation sweep/freeze failed' }
    & $projectPython -u 'src\evaluate_once.py'
    if ($LASTEXITCODE -ne 0) { throw 'Frozen DS2 evaluation failed' }
    & $projectPython -u 'src\openecg_demo.py'
    if ($LASTEXITCODE -ne 0) { throw 'Demo failed' }
    & $projectPython -u 'src\make_figures.py'
    if ($LASTEXITCODE -ne 0) { throw 'Figure generation failed' }
    Write-Output 'STEP_7_READY: training, frozen DS2 evaluation, noise, demo and figures complete; report rewrite awaits user confirmation.'
} finally {
    Stop-Transcript
}
