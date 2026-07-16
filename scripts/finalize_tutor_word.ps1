param(
    [Parameter(Mandatory = $true)]
    [string]$Docx,
    [Parameter(Mandatory = $false)]
    [string]$Pdf
)

$ErrorActionPreference = "Stop"
$resolvedDocx = (Resolve-Path -LiteralPath $Docx).Path
$appendixWord = "Ap$([char]0x00E9)ndice"
$prefixes = @(
    'Tabla 1:',
    'Tabla 2.',
    'Tabla 3.',
    'Tabla 7.1',
    'Tabla 7.2',
    'Tabla 7.3',
    'Tabla 7.4',
    'Tabla 7.5',
    'Tabla A.1',
    'Figura 1.',
    'Figura 7.1',
    'Figura 7.2',
    'Figura 7.3',
    'Figura 7.4',
    'Figura 7.5',
    'Figura 7.6',
    'Figura 7.7',
    'Figura A.1',
    'Figura A.2',
    "$appendixWord A.",
    "$appendixWord B.",
    "$appendixWord C."
)

function Get-ParagraphText {
    param([Parameter(Mandatory = $true)][object]$Paragraph)
    return $Paragraph.Range.Text.Trim([char]13, [char]7, ' ')
}

function Find-BodyStart {
    param([Parameter(Mandatory = $true)][object]$Document)
    $start = $null
    foreach ($paragraph in $Document.Paragraphs) {
        $text = Get-ParagraphText $paragraph
        if ($text.StartsWith('INTRODUCCI', [System.StringComparison]::OrdinalIgnoreCase)) {
            $start = $paragraph.Range.Start
        }
    }
    if ($null -eq $start) {
        throw 'No se encontro el inicio del cuerpo.'
    }
    return $start
}

function Sync-StaticLists {
    param([Parameter(Mandatory = $true)][object]$Document)
    $bodyStart = Find-BodyStart $Document
    $entries = @{}
    $bodies = @{}
    foreach ($paragraph in $Document.Paragraphs) {
        $text = Get-ParagraphText $paragraph
        foreach ($prefix in $prefixes) {
            if ($text.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
                if ($paragraph.Range.Start -lt $bodyStart -and $text.Contains([char]9)) {
                    $entries[$prefix] = $paragraph
                }
                elseif (
                    $paragraph.Range.Start -ge $bodyStart -and
                    -not $text.Contains([char]9) -and
                    -not $bodies.ContainsKey($prefix)
                ) {
                    $bodies[$prefix] = $paragraph
                }
                break
            }
        }
    }
    foreach ($prefix in $prefixes) {
        if (-not $entries.ContainsKey($prefix) -or -not $bodies.ContainsKey($prefix)) {
            throw "No se pudo sincronizar '$prefix'."
        }
        $entry = $entries[$prefix]
        $body = $bodies[$prefix]
        $page = [int]$body.Range.Information(3)
        $replacement = "$(Get-ParagraphText $body)`t$page"
        $range = $entry.Range.Duplicate
        $range.End = $range.End - 1
        $range.Text = $replacement
        $entry.Range.ParagraphFormat.TabStops.ClearAll()
        [void]$entry.Range.ParagraphFormat.TabStops.Add(450, 2, 0)
    }
}

$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($resolvedDocx)

    for ($pass = 1; $pass -le 2; $pass++) {
        foreach ($toc in $document.TablesOfContents) { $toc.Update() }
        foreach ($field in $document.Fields) { [void]$field.Update() }
        $document.Repaginate()
        Sync-StaticLists $document
    }

    $document.Repaginate()
    $document.Save()
    if ($Pdf) {
        $resolvedPdf = [System.IO.Path]::GetFullPath($Pdf)
        [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($resolvedPdf)) | Out-Null
        $document.ExportAsFixedFormat($resolvedPdf, 17)
    }
    $pages = $document.ComputeStatistics(2)
    Write-Output "PAGES=$pages"
}
finally {
    if ($null -ne $document) { $document.Close($false) }
    if ($null -ne $word) { $word.Quit() }
    if ($null -ne $document) {
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($document)
    }
    if ($null -ne $word) {
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
