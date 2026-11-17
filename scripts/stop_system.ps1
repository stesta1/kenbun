<#
.SYNOPSIS
  Corta los procesos que levantó scripts\start_system.ps1. Los contenedores de Docker quedan
  corriendo; para pararlos: docker compose stop.
#>
$root = Split-Path $PSScriptRoot -Parent
$pidFile = Join-Path $root "logs\pids.json"
if (-not (Test-Path $pidFile)) { Write-Host "No hay un sistema arrancado por start_system.ps1."; return }

$processes = Get-Content $pidFile -Raw | ConvertFrom-Json
foreach ($entry in $processes.PSObject.Properties) {
    $running = Get-Process -Id $entry.Value -ErrorAction SilentlyContinue
    if ($running) {
        taskkill /PID $entry.Value /T /F | Out-Null
        Write-Host ("Cortado {0} (PID {1})" -f $entry.Name, $entry.Value)
    } else {
        Write-Host ("{0} ya no estaba corriendo" -f $entry.Name)
    }
}
Remove-Item $pidFile
