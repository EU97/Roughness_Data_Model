# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,line-too-long,wrong-import-position
"""
Procesa en lote todos los grupos y especímenes bajo una carpeta raíz.
Usa src/Single.py (función procesar_carpeta) para cada espécimen.
"""
import os
import sys
import json
import csv
import argparse
from typing import List, Dict

# Permitir import relativo del módulo Single.py
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from Single import procesar_carpeta  # noqa: E402


def find_specimen_folders(root: str) -> List[str]:
    """Encuentra carpetas con archivos *.tx1/*.tx2/*.tx3 bajo root.
    Regresa lista de carpetas (paths absolutos).
    """
    matches: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        lower = {f.lower() for f in filenames}
        if any(f.endswith('.tx1') for f in lower) and any(f.endswith('.tx2') for f in lower) and any(f.endswith('.tx3') for f in lower):
            matches.append(os.path.abspath(dirpath))
    return matches


def main():
    """CLI de procesamiento en lote para rugosidad."""
    parser = argparse.ArgumentParser(description='Procesamiento en lote de rugosidad (ISO 4287/13565-2).')
    parser.add_argument('root', nargs='?', default=None, help='Carpeta raíz con subcarpetas de especímenes (por defecto data/).')
    parser.add_argument('--apply-filter', action='store_true', help='Aplicar filtrado ISO 16610 en cada espécimen.')
    parser.add_argument('--cutoff-mm', type=float, default=0.8, help='Longitud de corte (λc) en mm para el filtro Gaussiano.')
    parser.add_argument('--filter-source', choices=['primary', 'roughness'], default='primary', help='Fuente para filtrado.')
    parser.add_argument('--summary', default='batch_summary.json', help='Archivo JSON con resumen de resultados.')
    parser.add_argument('--summary-csv', default='batch_summary.csv', help='Archivo CSV con resumen de resultados (compatible con Excel).')
    args = parser.parse_args()

    default_root = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'data'))
    root_dir = os.path.abspath(args.root) if args.root else default_root

    if not os.path.isdir(root_dir):
        print(f"Error: La carpeta raíz '{root_dir}' no existe.")
        sys.exit(1)

    specimen_dirs = sorted(find_specimen_folders(root_dir))
    if not specimen_dirs:
        print('No se encontraron carpetas de especímenes con *.tx1/*.tx2/*.tx3.')
        sys.exit(2)

    summary: List[Dict] = []
    failures: List[Dict] = []

    for idx, folder in enumerate(specimen_dirs, start=1):
        rel = os.path.relpath(folder, root_dir)
        print(f"[{idx}/{len(specimen_dirs)}] Procesando: {rel}")
        try:
            result = procesar_carpeta(folder, apply_filter=args.apply_filter, cutoff_mm=args.cutoff_mm, filter_source=args.filter_source)
            # Reducir a campos útiles para el resumen
            summary.append({
                'folder': rel,
                'csv': os.path.relpath(result.get('csv_path', ''), root_dir),
                'Ra': result.get('Ra'),
                'Rq': result.get('Rq'),
                'Rz_ISO': result.get('Rz_ISO'),
                'RSm': result.get('RSm'),
                'Rpk': result.get('Rpk'),
                'Rk': result.get('Rk'),
                'Rvk': result.get('Rvk'),
                'Mr1': result.get('Mr1'),
                'Mr2': result.get('Mr2'),
            })
        except (FileNotFoundError, ValueError, OSError) as exc:  # Registro pero continuo con el resto
            failures.append({'folder': rel, 'error': str(exc)})
            print(f"  Error en '{rel}': {exc}")

    # Guardar resumen JSON
    out_path = os.path.abspath(os.path.join(root_dir, args.summary))
    try:
        with open(out_path, 'w', encoding='utf-8') as fh:
            json.dump({'summary': summary, 'failures': failures}, fh, ensure_ascii=False, indent=2)
        print(f"Resumen guardado en: {out_path}")
    except OSError as exc:
        print(f"No se pudo guardar el resumen: {exc}")

    # Guardar resumen CSV (UTF-8 con BOM para Excel)
    out_csv_path = os.path.abspath(os.path.join(root_dir, args.summary_csv))
    try:
        with open(out_csv_path, 'w', newline='', encoding='utf-8-sig') as fh:
            writer = csv.writer(fh)
            writer.writerow(['folder', 'csv', 'Ra', 'Rq', 'Rz_ISO', 'RSm', 'Rpk', 'Rk', 'Rvk', 'Mr1', 'Mr2'])
            for row in summary:
                writer.writerow([
                    row.get('folder', ''),
                    row.get('csv', ''),
                    row.get('Ra', ''),
                    row.get('Rq', ''),
                    row.get('Rz_ISO', ''),
                    row.get('RSm', ''),
                    row.get('Rpk', ''),
                    row.get('Rk', ''),
                    row.get('Rvk', ''),
                    row.get('Mr1', ''),
                    row.get('Mr2', ''),
                ])
        print(f"Resumen CSV guardado en: {out_csv_path}")
    except OSError as exc:
        print(f"No se pudo guardar el resumen CSV: {exc}")


if __name__ == '__main__':
    main()
