# Web App (GitHub Pages)

Esta carpeta contiene la app web estatica para ejecutarse sin backend.

## Publicacion en GitHub Pages

1. Ir a Settings > Pages.
2. Source: la rama de trabajo o rama principal.
3. Folder: `/docs`.
4. Guardar.

## Subdominio

1. En Pages > Custom domain, definir por ejemplo `tool.tudominio.com`.
2. Crear `docs/CNAME` con el valor exacto del subdominio.
3. En tu DNS, crear un registro CNAME:
- Host: `tool`
- Target: `<usuario>.github.io`
4. Activar `Enforce HTTPS` cuando la propagacion termine.

## Nota

La implementacion actual es estructura inicial (placeholder) para flujo local:

- Carga de ZIP.
- Organizacion de tripletas.
- Worker para procesamiento en segundo plano.
- Export simple de resultado JSON.

El port completo de formulas ISO desde Python a JavaScript se hara en siguientes iteraciones.
