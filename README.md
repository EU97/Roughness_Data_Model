# Roughness_Data_Model

Procesamiento de datos de rugosidad (ISO 4287:1997) con exportación CSV y gráficas.

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
python .\src\main.py
```

O pasar otra carpeta con archivos `*.tx1`, `*.tx2`, `*.tx3`:

```powershell
python .\src\main.py "c:\\ruta\\a\\carpeta"
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

## Notas

- Los parámetros Rk, Rpk, Rvk se estiman con un método heurístico. Para ISO 13565-2 se puede implementar como mejora futura.
