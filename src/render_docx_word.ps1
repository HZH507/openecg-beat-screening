param(
    [Parameter(Mandatory=$true)][string]$InputDocx,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [int]$Dpi = 150,
    [int[]]$Pages = @()
)
# Native Word page EMF -> raster PNG. No PDF/XPS export is performed.
# The source is opened read-only and closed without saving.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
function Get-PngHash([Drawing.Bitmap]$Image) {
    $memory = [IO.MemoryStream]::new()
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $Image.Save($memory,[Drawing.Imaging.ImageFormat]::Png)
        return ([BitConverter]::ToString($sha.ComputeHash($memory.ToArray()))).Replace('-','').ToLowerInvariant()
    } finally { $sha.Dispose(); $memory.Dispose() }
}
$inputPath = (Resolve-Path -LiteralPath $InputDocx).Path
$sourceHashBefore = (Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash
$outputPath = [IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
$wordApp = $null
$document = $null
try {
    $wordApp = New-Object -ComObject Word.Application
    $wordApp.Visible = $false
    $wordApp.DisplayAlerts = 0
    $wordApp.AutomationSecurity = 3
    $wordApp.ScreenUpdating = $true
    $document = $wordApp.Documents.Open($inputPath, $false, $true)
    $document.Repaginate()
    $document.ActiveWindow.View.Type = 3
    $pageCount = [int]$document.ComputeStatistics(2)
    $starts = @()
    for ($i=1; $i -le $pageCount; $i++) {
        $range = $document.GoTo(1,1,$i)
        $starts += [int]$range.Start
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($range)
    }
    $starts += [int]$document.Content.End
    $pageRows = @()
    $rendered = @()
    $renderPages = if ($Pages.Count -eq 0) { @(1..$pageCount) } else { @($Pages) }
    foreach ($number in $renderPages) {
        if ($number -lt 1 -or $number -gt $pageCount) { throw "Invalid page requested: $number" }
    }
    for ($i=1; $i -le $pageCount; $i++) {
        $range = $document.Range($starts[$i-1],$starts[$i])
        $plain = $range.Text -replace '[\x00-\x08\x0B\x0C\x0E-\x1F]',''
        $clean = ($plain -replace '\s+',' ').Trim()
        $pictures = [int]$range.InlineShapes.Count
        $positionRange = $document.Range($starts[$i-1],$starts[$i-1])
        $sectionNumber = [int]$positionRange.Information(2)
        $section = $document.Sections.Item($sectionNumber)
        $sectionStart = $document.Range($section.Range.Start,$section.Range.Start)
        $isFirstSectionPage = ([int]$sectionStart.Information(3) -eq $i)
        $headerKind = 1
        if ($isFirstSectionPage -and $section.PageSetup.DifferentFirstPageHeaderFooter) { $headerKind=2 }
        elseif ($section.PageSetup.OddAndEvenPagesHeaderFooter -and ([int]$positionRange.Information(1)%2 -eq 0)) { $headerKind=3 }
        $expectedHeader = ($section.Headers.Item($headerKind).Range.Text -replace '\s+',' ').Trim()
        $entry = [ordered]@{
            physical_page=$i
            adjusted_page=[int]$positionRange.Information(1)
            section=$sectionNumber
            expected_header_text=$expectedHeader
            start=$starts[$i-1]
            end=$starts[$i]
            text=$plain
            text_characters=$clean.Length
            inline_pictures=$pictures
            possibly_blank=($clean.Length -eq 0 -and $pictures -eq 0)
            first_text=$clean.Substring(0,[Math]::Min(90,$clean.Length))
            last_text=$clean.Substring([Math]::Max(0,$clean.Length-90))
        }
        if ($renderPages -contains $i) {
            # Word lazily paints repeated headers for off-screen pages. Select
            # and scroll to the page start before obtaining its EMF; otherwise
            # some later-page EMFs can omit the header even though it exists.
            $positionRange.Select()
            $document.ActiveWindow.ScrollIntoView($positionRange,$true)
            if ($expectedHeader.Length -gt 0) {
                # Explicitly enter/leave this page's header to invalidate the
                # display cache, then refresh in main-document view. This is
                # read-only and does not update or save the DOCX.
                $document.ActiveWindow.View.SeekView = 9
                $wordApp.ScreenRefresh()
                $document.ActiveWindow.View.SeekView = 0
                $positionRange.Select()
                $document.ActiveWindow.ScrollIntoView($positionRange,$true)
            }
            $wordApp.ScreenRefresh()
            $page = $document.ActiveWindow.Panes.Item(1).Pages.Item($i)
            [byte[]]$bits = $page.EnhMetaFileBits
            if ($bits.Length -lt 100) { throw "Word returned an empty EMF for page $i" }
            $stream = [IO.MemoryStream]::new($bits,$false)
            $metafile = [Drawing.Imaging.Metafile]::new($stream)
            $width = [int][Math]::Round([double]$section.PageSetup.PageWidth*$Dpi/72)
            $height = [int][Math]::Round([double]$section.PageSetup.PageHeight*$Dpi/72)
            $bitmap = [Drawing.Bitmap]::new($width,$height)
            $bitmap.SetResolution($Dpi,$Dpi)
            $graphics = [Drawing.Graphics]::FromImage($bitmap)
            try {
                $graphics.Clear([Drawing.Color]::White)
                $graphics.SmoothingMode = [Drawing.Drawing2D.SmoothingMode]::HighQuality
                $graphics.InterpolationMode = [Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
                $graphics.TextRenderingHint = [Drawing.Text.TextRenderingHint]::AntiAliasGridFit
                $graphics.DrawImage($metafile,0,0,$width,$height)
                $png = Join-Path $outputPath ('page-{0:D2}.png' -f $i)
                $bitmap.Save($png,[Drawing.Imaging.ImageFormat]::Png)
                # Audit the actual rendered header band as pixels. All body
                # pages in this report share the same header; its strip hash
                # can therefore be compared even when previews are downsampled.
                $headerY = [int][Math]::Round([double]$section.PageSetup.HeaderDistance*$Dpi/72)
                $bandStart = [Math]::Max(0,$headerY-[int][Math]::Round($Dpi*0.13))
                $bandHeight = [Math]::Min($height-$bandStart,[int][Math]::Round($Dpi*0.40))
                $band = $bitmap.Clone([Drawing.Rectangle]::new(0,$bandStart,$width,$bandHeight),[Drawing.Imaging.PixelFormat]::Format24bppRgb)
                $band.SetResolution($Dpi,$Dpi)
                $blankBand = [Drawing.Bitmap]::new($width,$bandHeight,[Drawing.Imaging.PixelFormat]::Format24bppRgb)
                $blankBand.SetResolution($Dpi,$Dpi)
                $blankGraphics = [Drawing.Graphics]::FromImage($blankBand)
                $blankGraphics.Clear([Drawing.Color]::White)
                $stripHash = Get-PngHash $band
                $entry.header_strip_sha256=$stripHash
                $entry.header_strip_nonblank=($stripHash -ne (Get-PngHash $blankBand))
                $entry.header_refresh='SeekView current header then main document; ScreenRefresh'
                $blankGraphics.Dispose();$blankBand.Dispose();$band.Dispose()
                $entry.png=$png
                $entry.emf_bytes=$bits.Length
                $entry.width_px=$width
                $entry.height_px=$height
                $rendered += $i
            } finally {
                $graphics.Dispose()
                $bitmap.Dispose()
                $metafile.Dispose()
                $stream.Dispose()
                [void][Runtime.InteropServices.Marshal]::ReleaseComObject($page)
            }
        }
        $pageRows += [pscustomobject]$entry
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($positionRange)
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($sectionStart)
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($range)
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($section)
    }
    $tableRows = @()
    for ($i=1; $i -le $document.Tables.Count; $i++) {
        $table = $document.Tables.Item($i)
        $startRange = $document.Range($table.Range.Start,$table.Range.Start)
        $endPosition = [Math]::Max($table.Range.Start,$table.Range.End-1)
        $endRange = $document.Range($endPosition,$endPosition)
        $startPage = [int]$startRange.Information(3)
        $endPage = [int]$endRange.Information(3)
        $tableRows += [pscustomobject]@{
            table=$i;start_page=$startPage;end_page=$endPage
            spans_pages=($startPage -ne $endPage)
            rows=[int]$table.Rows.Count;columns=[int]$table.Columns.Count
            first_row=($table.Rows.Item(1).Range.Text -replace '[\x07\r]',' | ').Trim()
        }
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($startRange)
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($endRange)
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($table)
    }
    $splitParagraphs = @()
    $orphanHeadings = @()
    for ($i=1; $i -le $document.Paragraphs.Count; $i++) {
        $paragraph = $document.Paragraphs.Item($i)
        $start = [int]$paragraph.Range.Start
        $end = [Math]::Max($start,[int]$paragraph.Range.End-1)
        $startRange = $document.Range($start,$start)
        $endRange = $document.Range($end,$end)
        $startPage = [int]$startRange.Information(3)
        $endPage = [int]$endRange.Information(3)
        $text = ($paragraph.Range.Text -replace '[\r\x07]','').Trim()
        if ($startPage -ne $endPage -and $text.Length -gt 0) {
            $firstFragment = $document.Range($start,[Math]::Min($end,$starts[$startPage]))
            $lastFragment = $document.Range([Math]::Max($start,$starts[$endPage-1]),$end)
            $firstLines = [int]$firstFragment.ComputeStatistics(1)
            $lastLines = [int]$lastFragment.ComputeStatistics(1)
            $splitParagraphs += [pscustomobject]@{
                paragraph=$i;start_page=$startPage;end_page=$endPage
                first_fragment_lines=$firstLines;last_fragment_lines=$lastLines
                possible_widow_or_orphan=($firstLines -le 1 -or $lastLines -le 1)
                text=$text
            }
            [void][Runtime.InteropServices.Marshal]::ReleaseComObject($firstFragment)
            [void][Runtime.InteropServices.Marshal]::ReleaseComObject($lastFragment)
        }
        if ($paragraph.OutlineLevel -lt 10 -and $i -lt $document.Paragraphs.Count) {
            $next = $document.Paragraphs.Item($i+1)
            $nextRange = $document.Range($next.Range.Start,$next.Range.Start)
            $nextPage = [int]$nextRange.Information(3)
            if ($endPage -lt $nextPage -and $text.Length -gt 0) {
                $orphanHeadings += [pscustomobject]@{paragraph=$i;page=$endPage;next_page=$nextPage;text=$text}
            }
            [void][Runtime.InteropServices.Marshal]::ReleaseComObject($nextRange)
            [void][Runtime.InteropServices.Marshal]::ReleaseComObject($next)
        }
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($startRange)
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($endRange)
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($paragraph)
    }
    $sections = @()
    for ($i=1; $i -le $document.Sections.Count; $i++) {
        $section = $document.Sections.Item($i)
        $sections += [pscustomobject]@{
            section=$i
            width_pt=[double]$section.PageSetup.PageWidth
            height_pt=[double]$section.PageSetup.PageHeight
            different_first_page=[bool]$section.PageSetup.DifferentFirstPageHeaderFooter
            header_text=$section.Headers.Item(1).Range.Text
            footer_text=$section.Footers.Item(1).Range.Text
            first_page_header_text=$section.Headers.Item(2).Range.Text
            first_page_footer_text=$section.Footers.Item(2).Range.Text
            primary_footer_field_codes=@($section.Footers.Item(1).Range.Fields | ForEach-Object {$_.Code.Text})
        }
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($section)
    }
    $diagnostics = [ordered]@{
        source=$inputPath;read_only=$true;renderer='Word Page.EnhMetaFileBits to System.Drawing PNG'
        exported_pdf=$false;page_count=$pageCount;rendered_pages=$rendered;dpi=$Dpi
        pages=$pageRows;tables=$tableRows;split_paragraphs=$splitParagraphs
        possible_widows_or_orphans=@($splitParagraphs | Where-Object {$_.possible_widow_or_orphan})
        orphan_headings=$orphanHeadings
        pages_with_missing_header=@($pageRows | Where-Object {$_.expected_header_text.Length -gt 0 -and $_.png -and -not $_.header_strip_nonblank} | ForEach-Object {$_.physical_page})
        blank_pages=@($pageRows | Where-Object {$_.possibly_blank} | ForEach-Object {$_.physical_page})
        sections=$sections;source_sha256=$sourceHashBefore
    }
    $diagnostics | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $outputPath 'word_layout_diagnostics.json') -Encoding utf8
    Write-Output ('Rendered '+$rendered.Count+' of '+$pageCount+' pages to '+$outputPath)
    Write-Output ('Tables spanning pages: '+@($tableRows | Where-Object {$_.spans_pages}).Count)
    Write-Output ('Blank pages: '+@($diagnostics.blank_pages).Count)
    Write-Output ('Rendered pages with missing expected header: '+@($diagnostics.pages_with_missing_header).Count)
    Write-Output ('Possible widow/orphan paragraphs: '+@($diagnostics.possible_widows_or_orphans).Count)
} finally {
    if ($null -ne $document) {
        $document.Close(0)
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($document)
    }
    if ($null -ne $wordApp) {
        $wordApp.Quit(0)
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($wordApp)
    }
}
$sourceHashAfter = (Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash
if ($sourceHashAfter -ne $sourceHashBefore) { throw 'Read-only source hash changed unexpectedly' }
