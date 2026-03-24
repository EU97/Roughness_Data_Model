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

# 4) Organizar archivos .tx sueltos en estructura Grupo/Espe
python .\src\Organize.py raw_data/ data/ --dry-run

# 5) Comparativas + PDFs
python .\src\Compare.py .\data --output-dir reports \
	--metrics Ra Rq Rz_ISO Rt RSm Rdq Rda Pc Rpk Rk Rvk Mr1 Mr2 --rank-metric Ra \
	--make-pdf --make-pdf-filtered --pdf-top-k 2

# 6) Dashboard interactivo (Streamlit)
streamlit run src/Dashboard.py
```

Salidas clave: CSV/PNG por espécimen en cada carpeta; `batch_summary.json/.csv` en `data/`; reportes y gráficos en `data/reports/` y PDFs opcionales.

## Referencia de Parámetros

| Parámetro | Norma | Descripción | Unidad |
|-----------|-------|-------------|--------|
| Ra | ISO 4287 §4.2.1 | Promedio aritmético de las alturas absolutas del perfil | µm |
| Rq | ISO 4287 §4.2.2 | Raíz del promedio cuadrático (RMS) de las alturas | µm |
| Rp | ISO 4287 §4.2.4 | Altura máxima de pico | µm |
| Rv | ISO 4287 §4.2.5 | Profundidad máxima de valle | µm |
| Rt | ISO 4287 §4.2.6 | Altura total del perfil (Rp − Rv) | µm |
| Rz | ISO 4287 §4.1.3 | Media de las amplitudes pico–valle máximas de 5 segmentos | µm |
| Rsk | ISO 4287 §4.3.1 | Asimetría (skewness) de la distribución de alturas | — |
| Rku | ISO 4287 §4.3.2 | Curtosis (kurtosis) de la distribución de alturas | — |
| RSm | ISO 4287 §4.3.3 | Anchura media de los elementos del perfil (cruces de cero) | µm |
| Rdq | ISO 4287 §4.4.2 | Pendiente RMS del perfil | µm/mm |
| Rda | ISO 4287 §4.4.1 | Pendiente media absoluta del perfil | µm/mm |
| Pc | ISO 4287 | Conteo de picos por milímetro sobre la línea media | 1/mm |
| Rk | ISO 13565-2 | Profundidad del núcleo funcional (curva de Abbott) | µm |
| Rpk | ISO 13565-2 | Altura reducida de picos (zona de desgaste inicial) | µm |
| Rvk | ISO 13565-2 | Profundidad reducida de valles (retención de lubricante) | µm |
| Mr1 | ISO 13565-2 | Porcentaje de material en el límite superior del núcleo | % |
| Mr2 | ISO 13565-2 | Porcentaje de material en el límite inferior del núcleo | % |

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
python .\src\Compare.py .\data --output-dir reports --metrics Ra Rq Rz_ISO Rt RSm Rdq Rda Pc Rpk Rk Rvk Mr1 Mr2 --rank-metric Ra
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

### Organizar archivos (Organize.py)

Organiza archivos `.tx1/.tx2/.tx3` sueltos (o en subcarpetas) en la estructura estándar `Grupo/Espe` que esperan los demás scripts.

```powershell
# Vista previa (no modifica nada)
python .\src\Organize.py raw_data/ data/ --dry-run

# Copiar (por defecto) con 5 especímenes por grupo
python .\src\Organize.py raw_data/ data/

# Mover en vez de copiar
python .\src\Organize.py raw_data/ data/ --move

# 3 especímenes por grupo
python .\src\Organize.py raw_data/ data/ --specimens-per-group 3

# Todo en un solo grupo
python .\src\Organize.py raw_data/ data/ --single-group
```

Parámetros:
- `source`: directorio origen con archivos `.tx`.
- `dest`: directorio destino (por defecto `data`).
- `--move`: mover en vez de copiar.
- `--dry-run`: mostrar acciones sin ejecutarlas.
- `--specimens-per-group N`: especímenes por grupo (defecto: 5).
- `--single-group`: agrupar todo en un solo grupo.
- `--prefix`: prefijo de carpeta de grupo (defecto: `Grupo`).

### Dashboard interactivo (Dashboard.py)

Dashboard Streamlit con tres páginas:
- **Single Specimen**: análisis interactivo con gráficos Plotly y slider de filtro λc.
- **Batch Overview**: tabla ordenable y heatmap de parámetros normalizados.
- **Group Comparison**: boxplots, barras media±std y tabla de estadísticas por grupo.

```powershell
pip install streamlit plotly
streamlit run src/Dashboard.py
```

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

### Formatos de entrada soportados

El sistema detecta automáticamente el formato de los archivos `.tx1`/`.tx2` al leer la primera línea:

**Formato 1 — Surfcom con cabecera** (formato original):
- `3.tx1`: perfil primario (línea 1: longitud mm, línea 2: número de puntos, resto: alturas µm)
- `3.tx2`: perfil de rugosidad (mismo formato)
- `3.tx3`: configuración (tabulado, latin-1)

Ejemplo `.tx1`:
```
10.00000
28087
10.6720
10.6640
...
```

**Formato 2 — Dos columnas (X,Z)** (formato nuevo, sin cabecera):
- `.tx1`: perfil primario — cada línea contiene `X,Z` separados por coma (X en mm, Z en µm)
- `.tx2`: perfil de rugosidad (mismo formato)
- `.tx3`: configuración (tabulado, latin-1 — idéntico al formato original)

Ejemplo `.tx1`:
```
0.0000000,-24.0120
0.0002035,-24.0200
0.0004069,-24.0360
...
```

> **Nota:** No se necesita ningún flag extra. `Single.py`, `Batch.py` y `Dashboard.py` detectan el formato automáticamente y aplican el lector correspondiente. Ambos formatos producen las mismas salidas (CSV, PNG, métricas).

- Salidas (en la misma carpeta):
	- resultados_rugosidad.csv (UTF-8 con BOM)
	- perfil_primario.png, perfil_rugosidad.png
	- curva_portancia_Rk.png
	- perfil_primario_corr.png/.csv, perfil_rugosidad_corr.png/.csv (con corrección de pendiente)
	- (si se activa) curva_portancia_Rk_16610.png, perfil_rugosidad_16610.png/.csv

- Salidas del modo lote (en la raíz):
- `batch_summary.json`: lista de especímenes con `folder`, `csv`, `Ra`, `Rq`, `Rz_ISO`, `Rt`, `RSm`, `Rdq`, `Rda`, `Pc`, `Rpk`, `Rk`, `Rvk`, `Mr1`, `Mr2` y lista de `failures`.
	- batch_summary.csv: igual resumen en CSV (UTF-8 con BOM) apto para Excel.

## Metodología y normas empleadas

Esta herramienta implementa parámetros de rugosidad según:

- ISO 4287:1997 (parámetros de amplitud, espaciamiento y pendiente): Ra, Rq, Rp, Rv, Rt, Rz, Rsk, Rku, RSm, Rdq, Rda y Pc.
- ISO 13565-2 (superficies con picos y valles sobresalientes): Rk, Rpk, Rvk, Mr1, Mr2 mediante la curva de material portante (Abbott–Firestone).

### Lectura de datos

Se soportan dos formatos de entrada para `.tx1`/`.tx2`. La detección es automática (basada en si la primera línea contiene una coma):

**Formato Surfcom (con cabecera):**
- Línea 1: longitud de medición (mm)
- Línea 2: número de puntos
- Líneas siguientes: alturas (µm), una por línea
- El eje X se construye con `np.linspace(0, longitud, n_puntos)`

**Formato dos columnas (sin cabecera):**
- Cada línea: `X,Z` (X en mm, Z en µm), separados por coma
- El eje X se toma directamente de la primera columna (mayor precisión espacial)
- La longitud de medición se deriva de los datos: `x[-1] - x[0]`

Archivo `.tx3`: metadatos/tabulado (latin-1). Idéntico en ambos formatos.

### Cálculo de parámetros (ISO 4287:1997)

- Ra: promedio aritmético de |z|.
- Rq: raíz del promedio de z².
- Rp, Rv, Rt: pico máximo, valle máximo y altura total del perfil (Rp − Rv).
- Rz (ISO 4287:1997): se divide el perfil de rugosidad en 5 segmentos iguales y se promedian las amplitudes pico–valle máximas por segmento.
- Rsk (asimetría) y Rku (curtosis): sesgo y curtosis estadística sobre z.
- RSm: anchura media de elementos por cruces de cero del perfil centrado (se resta la media para robustez), convirtiendo de puntos a longitud usando el eje X (mm → µm).
- Rdq: pendiente RMS del perfil (raíz del promedio de las derivadas al cuadrado).
- Rda: pendiente media absoluta del perfil (promedio de |dz/dx|).
- Pc: conteo de picos por milímetro que superan la línea media del perfil.

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
