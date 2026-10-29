$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$projectRoot = 'D:\openecg（1）'
$projectPython = 'D:\openecg\.venv\Scripts\python.exe'
$experimentRoot = Join-Path $projectRoot 'results\workpoint_cv_v2'
Set-Location -LiteralPath $projectRoot
$activeFolds = @{}
$nextFold = 1
$completedFolds = @()
try {
    while ($nextFold -le 5 -or $activeFolds.Count -gt 0) {
        while ($nextFold -le 5 -and $activeFolds.Count -lt 2) {
            $foldRoot = Join-Path $experimentRoot ('fold_'+$nextFold)
            New-Item -ItemType Directory -Force -Path $foldRoot | Out-Null
            $scriptPath = Join-Path $projectRoot 'src\train_workpoint_fold.py'
            $arguments = @('-u',('"'+$scriptPath+'"'),'--fold',[string]$nextFold)
            $process = Start-Process -FilePath $projectPython -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $foldRoot 'training.log') -RedirectStandardError (Join-Path $foldRoot 'training_errors.log')
            $activeFolds[$nextFold] = $process
            Write-Output ('Started fold '+$nextFold+'; 2 CPU threads per fold')
            $nextFold++
        }
        foreach ($foldNumber in @($activeFolds.Keys)) {
            $process = $activeFolds[$foldNumber]
            $process.Refresh()
            if ($process.HasExited) {
                $process.WaitForExit()
                if ($process.ExitCode -ne 0) { throw ('Fold '+$foldNumber+' failed; see its training_errors.log') }
                $completedFolds += [int]$foldNumber
                $activeFolds.Remove($foldNumber)
                Write-Output ('Completed fold '+$foldNumber)
            }
        }
        [pscustomobject]@{stage='five-fold OOF training';active_folds=@($activeFolds.Keys | Sort-Object);completed_folds=@($completedFolds | Sort-Object)} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $experimentRoot 'progress.json') -Encoding utf8
        if ($activeFolds.Count -gt 0) { Start-Sleep -Seconds 5 }
    }
    Write-Output 'All five outer folds complete. Pooling OOF predictions and selecting working points.'
    & $projectPython -u 'src\freeze_workpoint.py'
    if ($LASTEXITCODE -ne 0) { throw 'OOF selection/final refit failed' }
    & $projectPython -u 'src\evaluate_workpoint_once.py'
    if ($LASTEXITCODE -ne 0) { throw 'Final DS2 evaluation failed' }
    & $projectPython -u 'src\verify_workpoint.py'
    if ($LASTEXITCODE -ne 0) { throw 'Workpoint verification failed' }
    [pscustomobject]@{stage='complete; awaiting user confirmation';completed_folds=@(1,2,3,4,5)} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $experimentRoot 'progress.json') -Encoding utf8
    Write-Output 'STEP_ONE_COMPLETE: stop for user confirmation; report untouched.'
} catch {
    foreach ($process in $activeFolds.Values) {
        $process.Refresh()
        if (-not $process.HasExited) { Stop-Process -Id $process.Id }
    }
    throw
}
