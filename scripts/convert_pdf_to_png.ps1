param(
    [Parameter(Mandatory = $false)]
    [string]$DatasetPath = 'Release_1',

    [Parameter(Mandatory = $false)]
    [string]$OutputPath,

    [Parameter(Mandatory = $false)]
    [string]$LogPath,

    [Parameter(Mandatory = $false)]
    [double]$MaxMb = 10,

    [Parameter(Mandatory = $false)]
    [int]$MaxDim = 1080,

    [Parameter(Mandatory = $false)]
    [switch]$CompressAfterRender,

    [Parameter(Mandatory = $false)]
    [int]$Parallel = [Math]::Min(6, [Environment]::ProcessorCount),

    [Parameter(Mandatory = $false)]
    [string]$PythonExe
)

$ErrorActionPreference = 'Continue'
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptRoot

function Resolve-PathFromRoot {
    param(
        [Parameter(Mandatory = $true)][string]$PathValue,
        [Parameter(Mandatory = $false)][switch]$MustExist
    )

    $candidate = $PathValue
    if (-not [System.IO.Path]::IsPathRooted($candidate)) {
        $candidate = Join-Path $projectRoot $candidate
    }

    if ($MustExist) {
        return (Resolve-Path -LiteralPath $candidate).Path
    }

    return $candidate
}

$src = Resolve-PathFromRoot -PathValue $DatasetPath -MustExist
if (-not (Test-Path -LiteralPath $src -PathType Container)) {
    throw "Dataset folder not found: $DatasetPath"
}

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $srcName = Split-Path -Leaf $src
    $srcParent = Split-Path -Parent $src
    $dst = Join-Path $srcParent ($srcName + '_PNG')
} else {
    $dst = Resolve-PathFromRoot -PathValue $OutputPath
}

if ([string]::IsNullOrWhiteSpace($LogPath)) {
    $log = Join-Path $projectRoot 'conversion_log.txt'
} else {
    $log = Resolve-PathFromRoot -PathValue $LogPath
}

$processPy = Join-Path $scriptRoot 'compress_png_max.py'
$compressAfterRenderEnabled = [bool]$CompressAfterRender
$parallel = [Math]::Max(1, $Parallel)
$pdfToPpm = (Get-Command pdftoppm -ErrorAction Stop).Source

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
        $pythonExe = $venvPython
    } else {
        $pythonExe = (Get-Command python -ErrorAction Stop).Source
    }
} else {
    $pythonExe = Resolve-PathFromRoot -PathValue $PythonExe -MustExist
}

function Write-Log {
    param([string]$Message)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Add-Content -LiteralPath $log -Value $line
    Write-Host $Message
}

function Get-PdfPageCount {
    param([string]$PdfPath)
    $info = & pdfinfo $PdfPath 2>&1 | ForEach-Object { "$_" }
    foreach ($line in $info) {
        if ($line -match '^Pages:\s+(\d+)') {
            return [int]$Matches[1]
        }
    }
    return 0
}

function Get-PdfPageSizePts {
    param([string]$PdfPath)
    $info = & pdfinfo $PdfPath 2>&1 | ForEach-Object { "$_" }
    foreach ($line in $info) {
        if ($line -match 'Page size:\s+(\d+(?:\.\d+)?)\s+x\s+(\d+(?:\.\d+)?)\s+pts') {
            return [double]$Matches[1], [double]$Matches[2]
        }
    }
    return 612.0, 792.0
}

function Get-PdfNativeDpi {
    param([string]$PdfPath)
    $list = & pdfimages -list $PdfPath 2>&1 | ForEach-Object { "$_" }
    $ppis = @()
    foreach ($line in $list) {
        if ($line -match '^\s*\d+\s+\d+\s+image\s+') {
            $cols = ($line -split '\s+') | Where-Object { $_ -ne '' }
            if ($cols.Count -ge 14) {
                $xPpi = [int]$cols[12]
                if ($xPpi -gt 0) { $ppis += $xPpi }
            }
        }
    }
    if ($ppis.Count -gt 0) {
        return ($ppis | Measure-Object -Maximum).Maximum
    }
    return 72
}

function Get-RenderDpi {
    param([string]$PdfPath)
    $native = Get-PdfNativeDpi -PdfPath $PdfPath
    $wPts, $hPts = Get-PdfPageSizePts -PdfPath $PdfPath
    $maxPts = [Math]::Max($wPts, $hPts)
    $dpi1080 = [int][Math]::Floor($maxDim * 72.0 / $maxPts)
    if ($dpi1080 -lt 72) { $dpi1080 = 72 }
    return [Math]::Min($native, $dpi1080)
}

function Test-PdfConverted {
    param([string]$BaseName, [int]$PageCount)
    if ($PageCount -le 0) { return $false }
    for ($page = 1; $page -le $PageCount; $page++) {
        $pagePath = Join-Path $dst ($BaseName + '-' + $page + '.png')
        if (-not (Test-Path -LiteralPath $pagePath)) { return $false }
    }
    return $true
}

function Get-MissingPages {
    param([string]$BaseName, [int]$PageCount)
    $missing = New-Object System.Collections.Generic.List[int]
    for ($page = 1; $page -le $PageCount; $page++) {
        $pagePath = Join-Path $dst ($BaseName + '-' + $page + '.png')
        if (-not (Test-Path -LiteralPath $pagePath)) {
            $missing.Add($page)
        }
    }
    return [int[]]$missing.ToArray()
}

if (-not (Test-Path $dst)) {
    New-Item -ItemType Directory -Path $dst | Out-Null
}

Write-Log "=== 1080p / ${maxMb}MB run ($parallel workers, skip existing) ==="

$pdfs = Get-ChildItem -LiteralPath $src -Filter '*.pdf' -File | Sort-Object Name
$total = $pdfs.Count
$skipped = 0
$failures = @()
$pending = New-Object System.Collections.Generic.List[object]
$index = 0

foreach ($pdf in $pdfs) {
    $index++
    $pageCount = Get-PdfPageCount -PdfPath $pdf.FullName

    if (Test-PdfConverted -BaseName $pdf.BaseName -PageCount $pageCount) {
        $skipped++
        Write-Log "[$index/$total] SKIP $($pdf.Name) (complete)"
        continue
    }

    $missingPages = @(Get-MissingPages -BaseName $pdf.BaseName -PageCount $pageCount)
    if ($missingPages.Count -eq 0) { continue }

    $dpi = Get-RenderDpi -PdfPath $pdf.FullName
    $pending.Add([pscustomobject]@{
        Index        = $index
        Name         = $pdf.Name
        Path         = $pdf.FullName
        BaseName     = $pdf.BaseName
        Dpi          = $dpi
        PageCount    = $pageCount
        MissingPages = $missingPages
    })
}

Write-Log "Pending PDFs: $($pending.Count) | Skipped complete: $skipped | Max ${maxDim}px @ ${maxMb}MB"

$converted = 0
$queue = [System.Collections.Queue]::new()
foreach ($item in $pending) { [void]$queue.Enqueue($item) }
$running = @()

while ($queue.Count -gt 0 -or $running.Count -gt 0) {
    while ($running.Count -lt $parallel -and $queue.Count -gt 0) {
        $item = $queue.Dequeue()
        $missCount = $item.MissingPages.Count
        Write-Log "[$($item.Index)/$total] START $($item.Name) @ $($item.Dpi)dpi ($missCount missing pages)"
        $job = Start-Job -ScriptBlock {
            param($PdfPath, $Dst, $BaseName, $Dpi, $MissingPagesCsv, $ProcessPy, $MaxMb, $MaxDim, $PdfToPpm, $PythonExe, $CompressAfterRender)
            $prefix = Join-Path $Dst $BaseName
            $saved = @()
            $pages = @()
            if ($MissingPagesCsv) {
                $pages = $MissingPagesCsv -split ',' | ForEach-Object { [int]$_ }
            }
            foreach ($page in $pages) {
                $null = & $PdfToPpm -png -r $Dpi -f $page -l $page $PdfPath $prefix 2>&1
                if ($LASTEXITCODE -ne 0) { return 1 }
                $png = Join-Path $Dst ($BaseName + '-' + $page + '.png')
                if (Test-Path -LiteralPath $png) { $saved += $png }
            }
            if ($CompressAfterRender -and $saved.Count -gt 0) {
                $null = & $PythonExe $ProcessPy --max-mb $MaxMb --max-dim $MaxDim --jobs 2 @saved 2>&1
                if ($LASTEXITCODE -ne 0) { return 2 }
            }
            return 0
        } -ArgumentList $item.Path, $dst, $item.BaseName, $item.Dpi, ($item.MissingPages -join ','), $processPy, $maxMb, $maxDim, $pdfToPpm, $pythonExe, $compressAfterRenderEnabled
        $running += [pscustomobject]@{ Job = $job; Item = $item }
    }

    if ($running.Count -eq 0) { break }

    $done = Wait-Job -Job ($running.Job) -Any
    $slot = $running | Where-Object { $_.Job.Id -eq $done.Id }
    $running = $running | Where-Object { $_.Job.Id -ne $done.Id }

    $code = Receive-Job -Job $done
    Remove-Job -Job $done

    if ($code -ne 0) {
        $failures += $slot.Item.Name
        Write-Log "[$($slot.Item.Index)/$total] FAILED $($slot.Item.Name) (code $code)"
    } else {
        $converted++
        Write-Log "[$($slot.Item.Index)/$total] DONE $($slot.Item.Name)"
    }
}

Write-Log ''
Write-Log 'Done.'
Write-Log "PDFs skipped (already complete): $skipped"
Write-Log "PDFs processed this run: $converted"
Write-Log "PDFs failed: $($failures.Count)"
if ($failures.Count -gt 0) {
    $failures | ForEach-Object { Write-Log "  FAILED $_" }
}
