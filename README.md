# Roughness_Data_Model

Procesamiento de datos de rugosidad (ISO 4287:1997) e ISO 13565-2 con exportación CSV y gráficas.

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
- Los cálculos se realizan sobre el perfil de rugosidad (.tx2). Para otras condiciones (filtros de ondulación conforme a ISO 16610), podrían añadirse etapas de filtrado previas.
