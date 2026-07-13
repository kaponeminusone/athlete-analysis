# Empaqueta pesos CNN de venue para subir a Google Drive.
# Uso:  .\hoplab-cloud\colab\pack_venue_model.ps1
# Salida: venue-default-weights.zip en la raíz del repo.

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot
python (Join-Path $PSScriptRoot "pack_venue_model.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host ""
Write-Host "Siguiente:"
Write-Host "  1) Sube venue-default-weights.zip a Drive"
Write-Host "  2) Descomprime en hoplab-data/venues/default/"
Write-Host "     (o en hoplab-data/venues-upload/)"
Write-Host "  3) En Colab, ejecuta la celda install_venue_model"
