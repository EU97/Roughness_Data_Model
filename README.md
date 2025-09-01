# Roughness_Data_Model

Procesamiento de datos de rugosidad (ISO 4287:1997) e ISO 13565-2 con exportación CSV y gráficas.

## Inicio rápido

```powershell
# 1) Crear/activar venv y deps
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2) Un espécimen (con filtrado ISO 16610 opcional)
python .\src\Single.py .\data\GrupoI\EspeI --apply-filter --cutoff-mm 0.8 --filter-source primary

# 3) Lote completo (data/)
python .\src\Batch.py .\data --apply-filter --cutoff-mm 0.8 --filter-source primary

# 4) Comparativas + PDFs
python .\src\Compare.py .\data --output-dir reports \
	--metrics Ra Rq Rz_ISO RSm Rpk Rk Rvk Mr1 Mr2 --rank-metric Ra \
	--make-pdf --make-pdf-filtered --pdf-top-k 2
```

Salidas clave: CSV/PNG por espécimen en cada carpeta; `batch_summary.json/.csv` en `data/`; reportes y gráficos en `data/reports/` y PDFs opcionales.

## Requisitos

- Python 3.10 o superior (probado con 3.13)

## Instalación (reproducible)

1) Crear un entorno virtual en el repo

Windows PowerShell

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

Linux/macOS (bash/zsh)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Instalar dependencias

```powershell
pip install -r requirements.txt
```

## Cómo ejecutar

Por defecto procesa `data/GrupoI/EspeI`:

```powershell
python .\src\Single.py
```

O pasar otra carpeta con archivos `*.tx1`, `*.tx2`, `*.tx3`:

```powershell
python .\src\Single.py "c:\\ruta\\a\\carpeta"
```

### Filtrado ISO 16610 (opcional)

Puede activar el filtrado gaussiano (ISO 16610) para obtener el perfil de rugosidad desde el primario y recalcular los parámetros (también genera `curva_portancia_Rk_16610.png` y `perfil_rugosidad_16610.*`).

```powershell
python .\src\Single.py "c:\\ruta\\a\\carpeta" --apply-filter --cutoff-mm 0.8 --filter-source primary
```

Parámetros:
- `--apply-filter`: activa el filtrado.
- `--cutoff-mm`: longitud de corte λc en mm (p. ej., 0.8, 2.5).
- `--filter-source`: `primary` (recomendado) o `roughness`.

### Procesamiento en lote

Para procesar todos los especímenes bajo `data/` (o una raíz que indique):

```powershell
python .\src\Batch.py            # usa .\data como raíz por defecto
python .\src\Batch.py "c:\\ruta\\a\\root" --apply-filter --cutoff-mm 0.8 --filter-source primary --summary batch_summary.json
```

El proceso en lote:
- Detecta carpetas que contienen `*.tx1`, `*.tx2`, `*.tx3`.
- Ejecuta `procesar_carpeta` (de `src/Single.py`) en cada una.
- Deja los CSV/figuras en cada carpeta de espécimen.
- Genera un resumen `batch_summary.json` con métricas clave por carpeta y los fallos (si los hay).
- Además exporta `batch_summary.csv` (UTF-8 con BOM) para abrirlo en Excel fácilmente.

### Comparativas y reportes (Compare.py)

A partir del resumen del lote (`batch_summary.json`), se generan comparativas por grupo y entre grupos, rankings y gráficas.

```powershell
python .\src\Compare.py .\data --output-dir reports --metrics Ra Rq Rz_ISO RSm Rpk Rk Rvk Mr1 Mr2 --rank-metric Ra
```

Genera en `data/reports`:
- `group_<Grupo>_summary.csv`: estadísticas por grupo (media, std, min, max) para cada métrica.
- `group_<Grupo>_ranking_<M>.csv` y `group_<Grupo>_ranking_<M>.png`: ranking por métrica `M` (por defecto `Ra` o la indicada con `--rank-metric`).
- `groups_comparison_means.csv`: tabla de medias por grupo y métrica.
- `groups_means_<M>.png`: barras con medias±std por grupo.
- `groups_boxplot_<M>.png`: boxplots por grupo.
- `report.md`: resumen en Markdown.

Parámetros clave:
- `--metrics`: lista de métricas a considerar en tablas/figuras.
- `--rank-metric`: métrica para ordenar los rankings/figuras de ranking.
- `--output-dir`: carpeta destino de reportes (por defecto `reports`).
- `--compute-if-missing`: si no existe `batch_summary.json`, puede calcularlo en caliente (usa `Single.procesar_carpeta`).

### PDF opcional de reportes y PDF filtrado ISO 16610

Se pueden generar dos PDFs:
- PDF de salidas: compila los rankings por grupo y las gráficas comparativas entre grupos.
- PDF filtrado ISO 16610: inserta imágenes de `perfil_rugosidad_16610.png` y `curva_portancia_Rk_16610.png` de los Top-K especímenes por grupo según la métrica de ranking.

```powershell
# Requiere haber ejecutado Batch/Compare y, para el PDF filtrado, Single/Batch con --apply-filter
python .\src\Compare.py .\data --output-dir reports \
	--metrics Ra Rq Rz_ISO RSm Rpk Rk Rvk Mr1 Mr2 --rank-metric Ra \
	--make-pdf --make-pdf-filtered --pdf-out report_outputs.pdf --pdf-filtered-out report_filtered_16610.pdf \
	--pdf-top-k 2
```

Notas:
- Las imágenes filtradas ISO 16610 se generan cuando se ejecuta `Single.py` o `Batch.py` con `--apply-filter`.
- Puede cambiar `--rank-metric` para priorizar otra métrica (p. ej., `Rk`).

## Entradas y salidas

- Entradas: archivos tipo Surfcom en la carpeta objetivo
	- 3.tx1: perfil primario (línea 1 longitud mm, línea 2 puntos, resto alturas µm)
	- 3.tx2: perfil de rugosidad (formato igual)
	- 3.tx3: configuración (tabulado, latin-1)

- Salidas (en la misma carpeta):
	- resultados_rugosidad.csv (UTF-8 con BOM)
	- perfil_primario.png, perfil_rugosidad.png
	- curva_portancia_Rk.png
	- perfil_primario_corr.png/.csv, perfil_rugosidad_corr.png/.csv (con corrección de pendiente)
	- (si se activa) curva_portancia_Rk_16610.png, perfil_rugosidad_16610.png/.csv

- Salidas del modo lote (en la raíz):
	- batch_summary.json: lista de especímenes con `folder`, `csv`, `Ra`, `Rq`, `Rz_ISO`, `RSm`, `Rpk`, `Rk`, `Rvk`, `Mr1`, `Mr2` y lista de `failures`.
	- batch_summary.csv: igual resumen en CSV (UTF-8 con BOM) apto para Excel.

## Metodología y normas empleadas

Esta herramienta implementa parámetros de rugosidad según:

- ISO 4287:1997 (parámetros de amplitud y espaciamiento): Ra, Rq, Rp, Rv, Rz/Rt, Rsk, Rku y RSm.
- ISO 13565-2 (superficies con picos y valles sobresalientes): Rk, Rpk, Rvk, Mr1, Mr2 mediante la curva de material portante (Abbott–Firestone).

### Lectura de datos

- Archivos .tx1 (perfil primario) y .tx2 (perfil de rugosidad) con cabecera:
  - Línea 1: longitud de medición (mm)
  - Línea 2: número de puntos
  - Líneas siguientes: alturas (µm)
- Archivo .tx3: metadatos/tabulado (latin-1). Opcional para longitud; si falta, se usa la cabecera de .tx1/.tx2.

### Cálculo de parámetros (ISO 4287:1997)

- Ra: promedio aritmético de |z|.
- Rq: raíz del promedio de z².
- Rp, Rv, Rz/Rt: pico máximo, valle máximo y altura total (Rp − Rv).
- Rz (ISO 97): se divide el perfil de rugosidad en 5 segmentos iguales y se promedian las amplitudes pico–valle máximas por segmento.
- Rsk (asimetría) y Rku (curtosis): sesgo y curtosis estadística sobre z.
- RSm: anchura media de elementos por cruces de cero del perfil, convirtiendo de puntos a longitud usando el eje X (mm → µm).

### Parámetros funcionales (ISO 13565-2)

1. Curva de material portante (MRC):
	- Se ordena el perfil z de mayor a menor y se mapea r en [0,100] %.
2. Núcleo funcional:
	- Se busca la ventana del 40 % del dominio con mejor ajuste lineal (mínimo RMSE) y se define la línea del núcleo z = a·r + b.
3. Intersecciones Mr1 y Mr2:
	- Se obtienen r1 y r2 por los cruces entre MRC y la línea del núcleo (interpolación lineal).
4. Cálculo de Rk, Rpk, Rvk:
	- Rk = z(r1) − z(r2) (espesor del núcleo).
	- Rpk = A1 / r1, donde A1 es el área positiva entre MRC y la línea del núcleo en [0, r1].
	- Rvk = A2 / (100 − r2), donde A2 es el área positiva entre la línea del núcleo y la MRC en [r2, 100].
5. Mr1, Mr2: se reportan como r1 y r2 en %.

Notas:
- Los cálculos se realizan sobre el perfil de rugosidad (.tx2). Con `--apply-filter` se obtiene rugosidad a partir del primario conforme a ISO 16610.
- Codificaciones: entradas en `latin-1` para `.tx*`; salidas CSV con BOM (`utf-8-sig`) para compatibilidad con Excel.
- `src/Single.py` expone la función `procesar_carpeta(path, ...)` para uso programático.

## Reporte del proyecto

Se incluye `REPORT.md` con:
- Cómo ejecutar (comandos clave),
- Quality gates (build/syntax, smoke tests, estilo),
- Cobertura de requisitos (normas y funciones implementadas).
