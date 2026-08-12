# Plan de Implementacion Web Sin Backend (GitHub Pages + Subdominio)

## Objetivo

Publicar una herramienta web accesible para cualquier usuario desde un subdominio, sin backend y sin instalaciones locales manuales, donde el procesamiento de archivos `.tx1/.tx2/.tx3` ocurra en el navegador del usuario.

Flujo objetivo:

1. Usuario abre el sitio web.
2. Sube un `.zip` con archivos desordenados.
3. La app organiza internamente tripletas (`tx1`, `tx2`, `tx3`).
4. Calcula parametros y genera visualizaciones en el dispositivo del usuario.
5. Permite descargar resultados (`csv`, `json`, imagenes y zip final).

## Alcance de Esta Rama

1. Definir arquitectura objetivo y fases de implementacion.
2. Iniciar estructura web estatica en `docs/` para publicar en GitHub Pages.
3. Preparar base para integracion con subdominio custom (`CNAME`).

## Arquitectura Propuesta

- Hosting: GitHub Pages (sin servidor propio).
- Dominio: subdominio custom (`tool.tudominio.com` por ejemplo).
- Frontend: HTML + CSS + JavaScript.
- Procesamiento: lado cliente (browser) + Web Worker para tareas pesadas.
- Almacenamiento: temporal en memoria del navegador.
- Exportacion: descarga de archivos generados desde el browser.

## Modulos Tecnicos (Cliente)

1. `zip`: lectura y extraccion en memoria de archivos de entrada.
2. `organizer`: deteccion de tripletas y construccion logica de grupos/especimenes.
3. `parser`: lectura de formatos soportados (`Surfcom header` y `X,Z`).
4. `metrics`: calculo de parametros de rugosidad (ISO 4287 e ISO 13565-2).
5. `charts`: render de perfiles y comparativas.
6. `export`: armado de `batch_summary.json/csv` y zip final descargable.
7. `ui-state`: estado de interfaz, progreso y manejo de errores.

## Integracion con Subdominio

### GitHub Pages

1. Publicar desde rama actual y carpeta `docs/`.
2. Activar custom domain en settings del repositorio.
3. Crear archivo `CNAME` en la raiz publicada (`docs/CNAME`) con el subdominio final.
4. Configurar DNS:
- Registro `CNAME` en el proveedor DNS.
- Host: subdominio (por ejemplo `tool`).
- Target: `eu97.github.io`.

### SSL

- Habilitar `Enforce HTTPS` en GitHub Pages una vez propagado DNS.

## Fases de Implementacion

### Fase 1 - Base Web (actual)

1. Estructura estatica inicial en `docs/`.
2. Pantalla principal con carga de ZIP y estado.
3. Worker base para pipeline offline.

### Fase 2 - Parser + Organizacion

1. Lectura segura del zip.
2. Deteccion de tripletas completas/incompletas.
3. Tabla de validacion de entrada.

### Fase 3 - Calculo de Parametros

1. Port de funciones clave de `src/Single.py` a JS.
2. Salida por especimen y resumen de lote.
3. Validacion contra casos de `example_test/`.

### Fase 4 - Reportes y Descarga

1. Graficos de perfiles y comparativas.
2. Export `batch_summary.json` y `batch_summary.csv`.
3. Descarga de paquete final en zip.

### Fase 5 - Hardening

1. Mejoras de UX para movil y desktop.
2. Limites de tamano y manejo de memoria.
3. Pruebas de regresion con datasets reales.

## Riesgos y Mitigaciones

1. Rendimiento en movil:
- Mitigar con Web Worker y recomendacion de PC para lotes grandes.

2. Memoria para zips grandes:
- Procesamiento por bloques cuando aplique y limites de tamano.

3. Consistencia numerica respecto a Python:
- Suite de validacion cruzada (Python vs JS) con tolerancias.

## Criterios de Exito

1. Usuario puede procesar un zip completo sin instalar software.
2. Resultado descargable incluye resumen y archivos derivados.
3. Sitio disponible por subdominio con HTTPS.
4. Diferencia numerica aceptable frente a implementacion Python.

## Entregables de Esta Iteracion

1. Documento de plan en esta rama.
2. Estructura inicial web en `docs/`.
3. Guia de integracion con subdominio (plantilla CNAME).