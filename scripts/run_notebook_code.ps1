param(
    [Parameter(Mandatory = $true)]
    [string]$Notebook
)

$ErrorActionPreference = 'Stop'
$notebookPath = (Resolve-Path -LiteralPath $Notebook).Path
$notebookJson = Get-Content -Raw -LiteralPath $notebookPath | ConvertFrom-Json
$scriptLines = [System.Collections.Generic.List[string]]::new()
$scriptLines.Add('using Base64')

for ($cellIndex = 0; $cellIndex -lt $notebookJson.cells.Count; $cellIndex++) {
    $cell = $notebookJson.cells[$cellIndex]
    if ($cell.cell_type -ne 'code') {
        continue
    }

    $source = $cell.source -join ''
    if ([string]::IsNullOrWhiteSpace($source)) {
        continue
    }

    $encodedSource = [Convert]::ToBase64String(
        [Text.Encoding]::UTF8.GetBytes($source)
    )
    $cellLabel = ($notebookPath + '#cell-' + $cellIndex).Replace('\', '/')
    $scriptLines.Add(('println("CODEX_CELL {0}")' -f $cellIndex))
    $scriptLines.Add(
        ('include_string(Main, String(base64decode("{0}")), "{1}")' -f $encodedSource, $cellLabel)
    )
}

$scriptLines.Add('println("CODEX_NOTEBOOK_OK")')
$juliaArguments = @('--project=.', '-')

($scriptLines -join "`n") | & julia @juliaArguments
exit $LASTEXITCODE
