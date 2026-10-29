param(
    [Parameter(Mandatory=$true)][string]$InputDocx,
    [Parameter(Mandatory=$true)][string]$OutputPdf,
    [Parameter(Mandatory=$true)][string]$WordSummaryPath
)
$ErrorActionPreference='Stop'
$inputPath=(Resolve-Path -LiteralPath $InputDocx).Path
$outputPath=[IO.Path]::GetFullPath($OutputPdf)
$summaryPath=[IO.Path]::GetFullPath($WordSummaryPath)
$sourceHash=(Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash.ToLowerInvariant()
$wordApp=$null
$document=$null
try {
    $wordApp=New-Object -ComObject Word.Application
    $wordApp.Visible=$false
    $wordApp.DisplayAlerts=0
    $wordApp.AutomationSecurity=3
    $document=$wordApp.Documents.Open($inputPath,$false,$true)
    $document.Repaginate()
    $pageCount=[int]$document.ComputeStatistics(2)
    $lastStart=$document.GoTo(1,1,$pageCount)
    $lastRange=$document.Range($lastStart.Start,$document.Content.End)
    $lastText=[string]$lastRange.Text
    $adjustedPage=[int]$lastStart.Information(1)
    $document.ExportAsFixedFormat($outputPath,17)
    [ordered]@{
        source_docx=$inputPath;source_docx_sha256=$sourceHash
        output_pdf=$outputPath;word_page_count=$pageCount
        word_last_adjusted_page=$adjustedPage;word_last_page_text=$lastText
        renderer='Microsoft Word';read_only=$true
        exported_at_utc=[DateTime]::UtcNow.ToString('o')
    }|ConvertTo-Json -Depth 5|Set-Content -LiteralPath $summaryPath -Encoding utf8
    Write-Output ('Word physical pages: '+$pageCount+'; last footer page: '+$adjustedPage)
    Write-Output ('Exported '+$outputPath)
} finally {
    if($null -ne $document){$document.Close(0);[void][Runtime.InteropServices.Marshal]::ReleaseComObject($document)}
    if($null -ne $wordApp){$wordApp.Quit(0);[void][Runtime.InteropServices.Marshal]::ReleaseComObject($wordApp)}
}
if((Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sourceHash){throw 'DOCX changed during read-only PDF export'}
