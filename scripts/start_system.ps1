<#
.SYNOPSIS
  Levanta el sistema completo: Postgres, MinIO y MediaMTX en Docker, y los cuatro módulos como
  procesos nativos (uno de visión por instancia).

.DESCRIPTION
  Requiere Docker Desktop corriendo, un servidor de LLM local (Ollama) y el entorno virtual
  instalado. Cada proceso escribe su salida en logs\<nombre>.log. Para cortar todo:
  scripts\stop_system.ps1.

.PARAMETER Instances
  Nombres de config\instances.yaml a levantar. Vacío = todas. Si instances.yaml no existe, se levanta
  una única instancia con todas las cámaras del catálogo.

.PARAMETER VlmDevices
  GPU de cada worker del VLM, uno por entrada (ej. cuda:0,cuda:1). Vacío = un worker en MU_VLM_DEVICE.
  Cada worker carga su propia copia del modelo.

.PARAMETER Cameras
  camera_id de config\cameras.yaml. Si se indica, ignora instances.yaml y levanta una sola instancia
  con esas cámaras (útil para videos de prueba).

.PARAMETER ShortClips
  Para videos de prueba cortos: cierra los clips 3 s después del disparo (30 s por defecto) y
  deduplica incidentes en 20 s (90 s por defecto). No usar con cámaras reales.

.EXAMPLE
  .\scripts\start_system.ps1 -Cameras cam-v4,cam-v6 -ShortClips
  .\scripts\start_system.ps1 -Instances vision-1
#>
param(
    [string[]]$Instances = @(),
    [string[]]$Cameras = @(),
    [string[]]$VlmDevices = @(),
    [switch]$ShortClips
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$python = Join-Path $root ".venv\Scripts\python.exe"
$logs = Join-Path $root "logs"
$apiPort = if ($env:MU_API_PORT) { $env:MU_API_PORT } else { "8000" }
$pidFile = Join-Path $logs "pids.json"

if (-not (Test-Path $python)) { throw "No encuentro el entorno virtual en $python" }
New-Item -ItemType Directory -Force $logs | Out-Null
if (Test-Path $pidFile) { throw "Ya hay un sistema arrancado (logs\pids.json). Corré scripts\stop_system.ps1 primero." }

Set-Location $root
if ($ShortClips) {
    $env:MU_CLIP_POST_EVENT_SECONDS = "3"
    $env:MU_EVENT_DEDUP_WINDOW_SECONDS = "20"
}

$Cameras = @($Cameras | ForEach-Object { $_ -split "," } | Where-Object { $_ })
$Instances = @($Instances | ForEach-Object { $_ -split "," } | Where-Object { $_ })
$VlmDevices = @($VlmDevices | ForEach-Object { $_ -split "," } | Where-Object { $_ })
if ($Cameras -and $Instances) { throw "Usá -Cameras o -Instances, no los dos" }
if (-not $Cameras) {
    $configured = @(& $python -m monitoreo_urbano.common.instances | Where-Object { $_ })
    if ($LASTEXITCODE -ne 0 -or -not $configured) { throw "No se pudieron leer las instancias (config\instances.yaml, config\cameras.yaml)" }
    if (-not $Instances) { $Instances = $configured }
    $unknown = @($Instances | Where-Object { $configured -notcontains $_ })
    if ($unknown) { throw "Instancias no definidas: $($unknown -join ', '). Disponibles: $($configured -join ', ')" }
}

Write-Host "== Docker (Postgres, MinIO, MediaMTX)"
docker compose up -d --wait postgres minio mediamtx
if ($LASTEXITCODE -ne 0) { throw "docker compose falló. ¿Está Docker Desktop corriendo?" }

$started = [ordered]@{}

function Start-Module([string]$Name, [string[]]$Arguments) {
    $log = Join-Path $logs "$Name.log"
    $process = Start-Process $python -ArgumentList (@("-u") + $Arguments) -WorkingDirectory $root `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    $script:started[$Name] = $process.Id
    $script:started | ConvertTo-Json | Set-Content $pidFile -Encoding utf8
    Write-Host ("   {0,-22} PID {1}" -f $Name, $process.Id)
    return $log
}

function Wait-Log([string]$Name, [string]$Log, [string]$Pattern, [int]$TimeoutSeconds) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if ((Test-Path $Log) -and (Select-String -Path $Log -Pattern $Pattern -Quiet)) { return }
        if (-not (Get-Process -Id $started[$Name] -ErrorAction SilentlyContinue)) {
            throw "$Name terminó antes de estar listo. Mirá $Log y $Log.err"
        }
        Start-Sleep -Seconds 2
    }
    throw "$Name no estuvo listo en $TimeoutSeconds s. Mirá $Log y $Log.err"
}

Write-Host "== Módulos"
$null = Start-Module "query_service" @("-m", "monitoreo_urbano.query_service.main")
$deadline = (Get-Date).AddSeconds(240)
while ($true) {
    try { if ((Invoke-WebRequest "http://127.0.0.1:$apiPort/health" -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) { break } } catch {}
    if (-not (Get-Process -Id $started["query_service"] -ErrorAction SilentlyContinue)) { throw "query_service terminó. Mirá logs\query_service.log y .err" }
    if ((Get-Date) -gt $deadline) { throw "query_service no respondió en 240 s" }
    Start-Sleep -Seconds 2
}

# Un worker por GPU indicada; los workers comparten la cola de Postgres y se reparten los eventos.
$vlmWorkers = [ordered]@{}
if ($VlmDevices) {
    for ($i = 0; $i -lt $VlmDevices.Count; $i++) {
        $vlmWorkers["vlm_describer_$($i + 1)"] = @("--device", $VlmDevices[$i])
    }
} else {
    $vlmWorkers["vlm_describer"] = @()
}
$vlmLogs = @{}
foreach ($worker in $vlmWorkers.Keys) {
    $vlmLogs[$worker] = Start-Module $worker (@("-m", "monitoreo_urbano.vlm_describer.main") + $vlmWorkers[$worker])
}
foreach ($worker in $vlmWorkers.Keys) { Wait-Log $worker $vlmLogs[$worker] "listo, esperando eventos" 300 }
$storeLog = Start-Module "event_store" @("-m", "monitoreo_urbano.event_store.main")
Wait-Log "event_store" $storeLog "listo, esperando eventos" 300

if ($Cameras) {
    $null = Start-Module "vision_adhoc" @("-m", "monitoreo_urbano.vision_pipeline.main", "--camera-id", ($Cameras -join ","))
} else {
    foreach ($instance in $Instances) {
        $null = Start-Module "vision_$instance" @("-m", "monitoreo_urbano.vision_pipeline.main", "--instance", $instance)
    }
}

Write-Host ""
Write-Host "Listo. Interfaz: http://127.0.0.1:$apiPort/"
if ($Cameras) { Write-Host "Cámaras: $($Cameras -join ', ')   (publicar video en rtsp://localhost:8554/<camera_id>)" }
else { Write-Host "Instancias: $($Instances -join ', ')   (publicar video en rtsp://localhost:8554/<camera_id>)" }
Write-Host "Logs en: $logs"
