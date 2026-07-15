param(
    [Parameter(Mandatory = $false)]
    [string]$Path = "Tesis Maestro Final.docx"
)

$ErrorActionPreference = "Stop"
$resolved = (Resolve-Path -LiteralPath $Path).Path
$word = $null
$document = $null

function Convert-PlaceholderToEquation {
    param(
        [Parameter(Mandatory = $true)] [object]$Document,
        [Parameter(Mandatory = $true)] [string]$Placeholder,
        [Parameter(Mandatory = $true)] [string]$LinearEquation
    )

    $range = $Document.Content.Duplicate
    $find = $range.Find
    $find.ClearFormatting()
    $find.Text = $Placeholder
    $find.Forward = $true
    $find.Wrap = 0
    if (-not $find.Execute()) {
        return
    }

    $range.Text = $LinearEquation
    $range.Font.Name = "Cambria Math"
    [void]$Document.OMaths.Add($range)
    if ($range.OMaths.Count -ne 1) {
        throw "Word did not create an equation for: $Placeholder"
    }
    $range.OMaths.Item(1).BuildUp()
    $range.ParagraphFormat.Alignment = 1
}

function Update-StaticCaptionLists {
    param([Parameter(Mandatory = $true)] [object]$Document)

    $captions = @{}
    $listParagraphs = New-Object System.Collections.ArrayList
    foreach ($paragraph in $Document.Paragraphs) {
        $range = $paragraph.Range.Duplicate
        $text = $range.Text.Trim([char]13, [char]7)
        if ($text -match '^(Tabla|Figura) ([0-9]+(?:\.[0-9]+)?)') {
            $key = "$($matches[1]) $($matches[2])"
            if ($text.Contains([char]9)) {
                [void]$listParagraphs.Add(@($paragraph, $key))
            }
            else {
                $captions[$key] = @($text, [int]$range.Information(1))
            }
        }
        elseif ($text -match '^(Ap.ndice) ([A-Z])\.') {
            $key = "$($matches[1]) $($matches[2])"
            if ($text.Contains([char]9)) {
                [void]$listParagraphs.Add(@($paragraph, $key))
            }
            else {
                $captions[$key] = @($text, [int]$range.Information(1))
            }
        }
    }

    foreach ($item in $listParagraphs) {
        $paragraph = $item[0]
        $key = $item[1]
        if (-not $captions.ContainsKey($key)) {
            throw "No caption found for static list entry: $key"
        }
        $replacement = "$($captions[$key][0])`t$($captions[$key][1])"
        $body = $paragraph.Range.Duplicate
        $body.End = $body.End - 1
        $body.Text = $replacement
        $paragraph.Range.ParagraphFormat.TabStops.ClearAll()
        [void]$paragraph.Range.ParagraphFormat.TabStops.Add(450, 2, 0)
    }
}

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($resolved)

    $delta = [char]0x0394
    $smallDelta = [char]0x03B4
    $alpha = [char]0x03B1
    $beta = [char]0x03B2
    $gamma = [char]0x03B3
    $epsilon = [char]0x03B5
    $sigma = [char]0x03C3
    $sum = [char]0x2211
    $hat = [char]0x0302

    $adf = "${delta}y_t = $alpha + ${beta}t + ${gamma}y_(t-1) + " +
        "${sum}_(i=1)^p ${smallDelta}_i ${delta}y_(t-i) + ${epsilon}_t"
    $kpss = "KPSS = 1/(T^2 ${sigma}${hat}^2) ${sum}_(t=1)^T S_t^2,  " +
        "S_t = ${sum}_(i=1)^t e_i"
    $mz = "P_(t+h) = $alpha + ${beta}P${hat}_(t+h|t) + ${epsilon}_(t+h)"

    Convert-PlaceholderToEquation $document "ADF_EQUATION_PLACEHOLDER" $adf
    Convert-PlaceholderToEquation $document "KPSS_EQUATION_PLACEHOLDER" $kpss
    Convert-PlaceholderToEquation $document "MZ_EQUATION_PLACEHOLDER" $mz

    foreach ($toc in $document.TablesOfContents) { $toc.Update() }
    foreach ($tof in $document.TablesOfFigures) { $tof.Update() }
    $document.Fields.Update() | Out-Null
    foreach ($section in $document.Sections) {
        foreach ($header in $section.Headers) { $header.Range.Fields.Update() | Out-Null }
        foreach ($footer in $section.Footers) { $footer.Range.Fields.Update() | Out-Null }
    }
    $document.Repaginate()
    Update-StaticCaptionLists $document
    $document.Repaginate()
    Update-StaticCaptionLists $document
    $document.Save()
    Write-Output "Equations and indexes updated: $resolved"
}
finally {
    if ($document -ne $null) { $document.Close($false) }
    if ($word -ne $null) { $word.Quit() }
    if ($document -ne $null) { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($document) }
    if ($word -ne $null) { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
