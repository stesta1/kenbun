<#
.SYNOPSIS
  Corrida nocturna de tres pruebas, una detrás de otra (todas usan la GPU). Todo queda en reentreno\corrida_noche\<fecha>\.

  1. Integridad de 100 eventos (OE2) con el flujo vigente. Levanta el sistema sobre una base y un bucket propios de la
     prueba (la base habitual no se toca), publica por RTSP los 74 videos de choque (54 de choques_dominio2 y 20 de
     choques_dominio) en las dos cámaras de medición, opera la cola como un operador y verifica 100 eventos al azar.
  2. Fidelidad de la descripción breve vigente sobre 30 videos de choque (los 20 del set 2 y 10 más, semilla 42), con el
     modelo de 2B y la configuración del sistema. Deja la planilla de revisión a ciegas para los autores.
  3. Falsos positivos de v6 en tránsito normal sobre 24 secuencias de UA-DETRAC.

  Si una prueba falla, se registra y se sigue con la siguiente. No hace commits ni modifica código versionado.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\reentreno_v10\corrida_noche.ps1
  (desde la raíz del repositorio). Para correr solo algunas: -Pruebas 2,3
#>
param([int[]]$Pruebas = @(1, 2, 3))

$ErrorActionPreference = "Continue"
$App = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Re = Join-Path $App "reentreno"
$stamp = Get-Date -Format "yyyyMMdd_HHmm"
$Out = Join-Path $Re "corrida_noche\$stamp"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
Start-Transcript -Path (Join-Path $Out "corrida.log") | Out-Null
$env:PYTHONIOENCODING = "utf-8"
$Py = Join-Path $App ".venv\Scripts\python.exe"
Set-Location $App
function Paso([string]$t) { Write-Host ("`n=== {0} {1}" -f (Get-Date -Format "HH:mm:ss"), $t) }
# Cada paso de Python escribe su salida completa en su propio log (la transcripción no captura programas externos).
function Py([string]$log, [string[]]$a) { & $Py -X utf8 @a *>> $log; Write-Host "  (código de salida $LASTEXITCODE; log: $log)" }

# ---------------------------------------------------------------- Reproducibilidad
Paso "Registro de reproducibilidad"
$rep = Join-Path $Out "reproducibilidad.txt"
"Fecha: $(Get-Date -Format o)" | Out-File $rep -Encoding utf8
"Pruebas: $($Pruebas -join ',')" | Out-File $rep -Append -Encoding utf8
"Código: rama $(git -C $App rev-parse --abbrev-ref HEAD), commit $(git -C $App rev-parse HEAD)" | Out-File $rep -Append -Encoding utf8
"Cambios sin commit en el código:" | Out-File $rep -Append -Encoding utf8
git -C $App status --short --untracked-files=no | Out-File $rep -Append -Encoding utf8
"Scripts de la corrida (SHA-256):" | Out-File $rep -Append -Encoding utf8
Get-ChildItem $PSScriptRoot -Filter *.p* | Get-FileHash -Algorithm SHA256 | ForEach-Object { "  $($_.Hash)  $(Split-Path $_.Path -Leaf)" } | Out-File $rep -Append -Encoding utf8
"Pesos del detector (SHA-256):" | Out-File $rep -Append -Encoding utf8
Get-FileHash -Algorithm SHA256 (Join-Path $App "data\models\yolo\baseline_v6.pt") | ForEach-Object { "  $($_.Hash)  baseline_v6.pt" } | Out-File $rep -Append -Encoding utf8
& $Py -c "import sys, torch, ultralytics, transformers, cv2; print('Python', sys.version.split()[0]); print('torch', torch.__version__, 'CUDA', torch.version.cuda); print('ultralytics', ultralytics.__version__); print('transformers', transformers.__version__); print('opencv', cv2.__version__)" | Out-File $rep -Append -Encoding utf8
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | Out-File $rep -Append -Encoding utf8
docker compose images | Out-File $rep -Append -Encoding utf8
"Semillas: integridad 42 (acciones y muestra), fidelidad 42 (10 videos agregados), UA-DETRAC 42 (12 secuencias agregadas)" | Out-File $rep -Append -Encoding utf8

# ---------------------------------------------------------------- 1. Integridad
if ($Pruebas -contains 1) {
    $dir = Join-Path $Out "integridad"; New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $db = "monitoreo_integridad_$stamp"
    try {
        Paso "1. Integridad: servicios de Docker"
        docker compose up -d
        for ($i = 0; $i -lt 30; $i++) { docker compose exec -T postgres pg_isready -U monitoreo | Out-Null; if ($LASTEXITCODE -eq 0) { break }; Start-Sleep 2 }
        docker compose exec -T postgres psql -U monitoreo -d monitoreo_urbano -c "CREATE DATABASE $db"
        $env:MU_DATABASE_URL = "postgresql://monitoreo:monitoreo@localhost:5432/$db"
        $env:MU_S3_BUCKET_CLIPS = "clips-integridad-" + $stamp.Replace("_", "-")
        $env:MU_CAMERAS_CONFIG_PATH = Join-Path $Re "medicion\cameras_medicion.yaml"
        Paso "1. Integridad: sistema (base $db)"
        # Un logs\pids.json de una sesión anterior impide arrancar; stop_system corta lo que siga vivo y lo borra.
        if (Test-Path (Join-Path $App "logs\pids.json")) { & (Join-Path $App "scripts\stop_system.ps1") }
        & (Join-Path $App "scripts\start_system.ps1") -Cameras lat-cam-1, lat-cam-2
        Paso "1. Integridad: 54 videos de choques_dominio2"
        Py (Join-Path $dir "publicacion_1.log") @("scripts\generate_corpus_choques.py", "--carpeta", "..\test_ui\videos\choques_dominio2", "--patron", "E*.mp4", "--no-fire", "--out", (Join-Path $dir "corrida_1.json"))
        Paso "1. Integridad: 20 videos de choques_dominio"
        Py (Join-Path $dir "publicacion_2.log") @("scripts\generate_corpus_choques.py", "--carpeta", "..\test_ui\videos\choques_dominio", "--patron", "D*.mp4", "--no-fire", "--out", (Join-Path $dir "corrida_2.json"))
        # Con la regla vigente de dos niveles salen unos 0,56 eventos por video: los 74 no alcanzan para la muestra de 100.
        # Se suman 140 videos de la partición de prueba de ACCIDENT (semilla 42; mapa en data\integridad_extra\mapa_videos.csv).
        Paso "1. Integridad: 140 videos de la partición de prueba de ACCIDENT"
        Copy-Item (Join-Path $App "data\integridad_extra\mapa_videos.csv") (Join-Path $dir "mapa_videos_extra.csv")
        Py (Join-Path $dir "publicacion_3.log") @("scripts\generate_corpus_choques.py", "--carpeta", "data\integridad_extra", "--patron", "X*.mp4", "--no-fire", "--out", (Join-Path $dir "corrida_3.json"))
        Paso "1. Integridad: cola, acciones del operador y verificación"
        Py (Join-Path $dir "verificacion.log") @("scripts\reentreno_v10\integridad_flujo_vigente.py", "--out-dir", $dir)
        Copy-Item (Join-Path $App "logs\*.log") $dir -ErrorAction SilentlyContinue
    } catch {
        Write-Host "ERROR en integridad: $_"
    } finally {
        & (Join-Path $App "scripts\stop_system.ps1")
        Remove-Item Env:MU_DATABASE_URL, Env:MU_S3_BUCKET_CLIPS, Env:MU_CAMERAS_CONFIG_PATH -ErrorAction SilentlyContinue
    }
}

# ---------------------------------------------------------------- 2. Fidelidad
if ($Pruebas -contains 2) {
    $dir = Join-Path $Out "fidelidad_30"; New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $videos = "C01_calle_dia_seco,C02_distribuidor_dia_seco,C04_autopista_dia_lluvia,C05_autopista_dia_seco,C09_autopista_dia_nieve,C10_autopista_dia_seco,C11_autopista_dia_seco,C13_autopista_noche_seco,C14_estacionamiento_dia_seco,C15_interseccion_semaforo_dia_seco,C16_interseccion_semaforo_dia_seco,C17_interseccion_semaforo_dia_seco,C19_interseccion_semaforo_dia_lluvia,C21_interseccion_semaforo_dia_seco,C22_interseccion_semaforo_dia_seco,C25_interseccion_semaforo_noche_seco,C28_interseccion_semaforo_noche_seco,C29_interseccion_dia_seco,C31_autopista_dia_lluvia,C32_interseccion_semaforo_dia_seco,C03_autopista_dia_seco,C06_autopista_dia_lluvia,C07_autopista_dia_seco,C08_autopista_dia_seco,C12_autopista_dia_seco,C24_interseccion_semaforo_dia_seco,C26_interseccion_semaforo_noche_seco,C30_interseccion_dia_seco,C33_autopista_dia_seco,C37_autopista_dia_seco"
    try {
        Paso "2. Fidelidad: descripción breve vigente (2B) sobre 30 videos"
        # Misma configuración que la descripción breve evaluada antes (set2_corta): prompts vigentes, 8 cuadros, 250.880 píxeles.
        Py (Join-Path $dir "descripciones.log") @("scripts\compare_descriptions.py", "--label", "corta_2b_30", "--model", "Qwen/Qwen2-VL-2B-Instruct", "--input-mode", "images", "--max-frames", "8", "--max-pixels", "250880", "--from-dir", "--only", $videos, "--out", (Join-Path $dir "desc_corta_2b_30.json"))
        Paso "2. Fidelidad: planilla de revisión a ciegas"
        Py (Join-Path $dir "planilla.log") @("scripts\prepare_description_review.py", "--variants", (Join-Path $dir "desc_corta_2b_30.json"), "--clips-rel", "../../../../../test_ui/videos/choques", "--key", "descripciones_30_$stamp", "--filename", "veredictos_fidelidad_30.json", "--out", (Join-Path $dir "revision"))
    } catch {
        Write-Host "ERROR en fidelidad: $_"
    }
}

# ---------------------------------------------------------------- 3. UA-DETRAC
if ($Pruebas -contains 3) {
    try {
        Paso "3. Falsos positivos de v6 en UA-DETRAC (24 secuencias)"
        New-Item -ItemType Directory -Force -Path (Join-Path $Out "detrac") | Out-Null
        Py (Join-Path $Out "detrac\detrac.log") @("scripts\reentreno_v10\falsos_positivos_detrac.py", "--out-dir", (Join-Path $Out "detrac"))
    } catch {
        Write-Host "ERROR en UA-DETRAC: $_"
    }
}

Paso "TERMINADO. Resultados en $Out"
Stop-Transcript | Out-Null
