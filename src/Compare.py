# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,line-too-long
"""
Comparador y generador de reportes:
- Resume métricas por grupo (promedio, desviación, min, max) y ranking de especímenes.
- Compara grupos entre sí con promedios por métrica.

Usa el resumen generado por Batch.py (batch_summary.json). Si no existe y se
indica --compute-if-missing, calcula en caliente usando procesar_carpeta.
"""
from __future__ import annotations

import os
import sys
import json
import csv
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    from Batch import find_specimen_folders  # reutilizar descubrimiento
except (ImportError, OSError):
    def find_specimen_folders(root: str) -> List[str]:
        matches: List[str] = []
        for dirpath, _, filenames in os.walk(root):
            lower = {f.lower() for f in filenames}
            if any(f.endswith('.tx1') for f in lower) and any(f.endswith('.tx2') for f in lower) and any(f.endswith('.tx3') for f in lower):
                matches.append(os.path.abspath(dirpath))
        return matches

try:
    from Single import procesar_carpeta
except (ImportError, OSError):
    procesar_carpeta = None  # type: ignore


def _load_summary(json_path: str) -> Dict:
    with open(json_path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def _parse_group_spec(root_dir: str, folder_abs: str) -> Tuple[str, str]:
    rel = os.path.relpath(folder_abs, root_dir)
    parts = rel.replace('\\', '/').split('/')
    group = parts[0] if len(parts) > 0 else 'UNKNOWN'
    spec = parts[1] if len(parts) > 1 else os.path.basename(folder_abs)
    return group, spec


def _to_float(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _agg_metrics(rows: List[Dict[str, object]], metrics: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        vals = [_to_float(r.get(m)) for r in rows]
        arr = np.array([v for v in vals if v is not None], dtype=float)
        if arr.size == 0:
            out[m] = {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
        else:
            out[m] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
            }
    return out


def compare(root_dir: str, summary_json: str, output_dir: str, metrics: List[str], rank_metric: Optional[str] = None,
            make_pdf: bool = False, make_pdf_filtered: bool = False, pdf_out: Optional[str] = None,
            pdf_filtered_out: Optional[str] = None, pdf_top_k: int = 3) -> Dict[str, object]:
    """Genera comparaciones por grupo y entre grupos a partir de un batch_summary.json.

    Crea CSVs por grupo (estadísticos y ranking), comparaciones entre grupos (medias),
    gráficos PNG (rankings, medias±std, boxplots) y un reporte Markdown.

    Retorna paths y estadísticas agregadas.
    """
    os.makedirs(output_dir, exist_ok=True)
    data = _load_summary(summary_json)
    rows: List[Dict[str, object]] = data.get('summary', [])

    # Enriquecer filas con group/spec
    enriched: List[Dict[str, object]] = []
    for r in rows:
        folder_rel = r.get('folder', '')
        folder_abs = os.path.abspath(os.path.join(root_dir, folder_rel))
        group, spec = _parse_group_spec(root_dir, folder_abs)
        rr = dict(r)
        rr['group'] = group
        rr['specimen'] = spec
        enriched.append(rr)

    # Agrupar por grupo
    groups: Dict[str, List[Dict[str, object]]] = {}
    for r in enriched:
        groups.setdefault(str(r['group']), []).append(r)

    # Normalizar rank_metric
    if not rank_metric or rank_metric not in metrics:
        rank_metric = 'Ra' if 'Ra' in metrics else metrics[0]

    # Guardar resúmenes por grupo y rankings
    per_group_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for gname, grows in groups.items():
        stats_g = _agg_metrics(grows, metrics)
        per_group_stats[gname] = stats_g

        # CSV: resumen estadístico del grupo
        out_csv = os.path.join(output_dir, f'group_{gname}_summary.csv')
        with open(out_csv, 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.writer(fh)
            w.writerow(['metric', 'mean', 'std', 'min', 'max'])
            for m in metrics:
                s = stats_g[m]
                w.writerow([m, f"{s['mean']:.6f}", f"{s['std']:.6f}", f"{s['min']:.6f}", f"{s['max']:.6f}"])

        # CSV: ranking de especímenes (por una métrica clave seleccionable)
        def _rank_key(row, _metric=rank_metric):
            val = _to_float(row.get(_metric))
            return (val is None, val if val is not None else float('inf'))
        ranked = sorted(grows, key=_rank_key)
        out_rank = os.path.join(output_dir, f'group_{gname}_ranking_{rank_metric}.csv')
        with open(out_rank, 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.writer(fh)
            w.writerow(['group', 'specimen'] + metrics)
            for r in ranked:
                w.writerow([r['group'], r['specimen']] + [r.get(m, '') for m in metrics])

        # Gráfico de ranking (top 10) para el grupo por rank_metric
        top = ranked[:10]
        spec_names = [str(r['specimen']) for r in top]
        spec_vals = [_to_float(r.get(rank_metric)) for r in top]
        spec_vals = [v for v in spec_vals if v is not None]
        if spec_vals:
            plt.figure(figsize=(8, max(4, 0.4 * len(spec_names))))
            y_pos = np.arange(len(spec_names))
            plt.barh(y_pos, spec_vals, color='#4C78A8')
            plt.yticks(y_pos, spec_names)
            plt.xlabel(rank_metric)
            plt.title(f'Ranking {rank_metric} - {gname} (Top {len(spec_vals)})')
            plt.gca().invert_yaxis()
            out_png = os.path.join(output_dir, f'group_{gname}_ranking_{rank_metric}.png')
            plt.tight_layout()
            plt.savefig(out_png)
            plt.close()

    # Comparación entre grupos: tabla con promedios por métrica
    out_groups_csv = os.path.join(output_dir, 'groups_comparison_means.csv')
    with open(out_groups_csv, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['group'] + metrics)
        for gname in sorted(per_group_stats.keys()):
            row = [gname]
            for m in metrics:
                row.append(f"{per_group_stats[gname][m]['mean']:.6f}")
            w.writerow(row)

    # Gráficos comparativos entre grupos por métrica
    sorted_groups = sorted(per_group_stats.keys())
    for m in metrics:
        means = [per_group_stats[g][m]['mean'] for g in sorted_groups]
        stds = [per_group_stats[g][m]['std'] for g in sorted_groups]
        # Barras con barras de error (medias ± std)
        plt.figure(figsize=(max(6, 1.5*len(sorted_groups)), 5))
        x = np.arange(len(sorted_groups))
        plt.bar(x, means, yerr=stds, capsize=4, color='#72B7B2', edgecolor='#2A5D5E')
        plt.xticks(x, sorted_groups, rotation=30, ha='right')
        plt.ylabel(m)
        plt.title(f'Medias por grupo - {m}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'groups_means_{m}.png'))
        plt.close()

        # Boxplot de valores por grupo
        series: List[List[float]] = []
        for g in sorted_groups:
            vals = [_to_float(r.get(m)) for r in groups[g]]
            series.append([v for v in vals if v is not None])
        if any(len(s) > 0 for s in series):
            plt.figure(figsize=(max(6, 1.5*len(sorted_groups)), 5))
            plt.boxplot(series, tick_labels=sorted_groups, showmeans=True)
            plt.ylabel(m)
            plt.title(f'Distribución por grupo - {m}')
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'groups_boxplot_{m}.png'))
            plt.close()

    # Reporte Markdown breve
    out_md = os.path.join(output_dir, 'report.md')
    with open(out_md, 'w', encoding='utf-8') as fh:
        fh.write("# Reporte de comparación de rugosidad\n\n")
        fh.write(f"Raíz: {root_dir}\n\n")
        fh.write("## Resumen por grupo\n\n")
        for gname in sorted(per_group_stats.keys()):
            fh.write(f"### {gname}\n\n")
            fh.write("Métricas (media ± std):\\n\n")
            for m in metrics:
                s = per_group_stats[gname][m]
                fh.write(f"- {m}: {s['mean']:.3f} ± {s['std']:.3f} (min {s['min']:.3f}, max {s['max']:.3f})\n")
            fh.write("\n")
        fh.write("## Comparación entre grupos (medias)\n\n")
        fh.write("Ver 'groups_comparison_means.csv'.\n")

    # PDFs opcionales
    results: Dict[str, object] = {}
    if make_pdf:
        pdf_path = os.path.join(output_dir, pdf_out or 'report_outputs.pdf')
        with PdfPages(pdf_path) as pdf:
            # Portada
            fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape-ish
            fig.text(0.5, 0.7, 'Reporte de salidas (comparativas)', ha='center', va='center', fontsize=18)
            fig.text(0.5, 0.6, f"Raíz: {root_dir}", ha='center', va='center', fontsize=10)
            fig.text(0.5, 0.55, f"Métrica de ranking: {rank_metric}", ha='center', va='center', fontsize=10)
            pdf.savefig(fig); plt.close(fig)

            # Incluir rankings por grupo
            for gname in sorted(groups.keys()):
                img = os.path.join(output_dir, f'group_{gname}_ranking_{rank_metric}.png')
                if os.path.exists(img):
                    fig = plt.figure(figsize=(11.69, 8.27))
                    ax = fig.add_subplot(111); ax.axis('off')
                    ax.set_title(f'Ranking {rank_metric} - {gname}')
                    image = plt.imread(img)
                    ax.imshow(image)
                    ax.set_aspect('equal')
                    pdf.savefig(fig); plt.close(fig)

            # Incluir comparativas por métrica (means y boxplot)
            for m in metrics:
                for prefix in [f'groups_means_{m}.png', f'groups_boxplot_{m}.png']:
                    img = os.path.join(output_dir, prefix)
                    if os.path.exists(img):
                        fig = plt.figure(figsize=(11.69, 8.27))
                        ax = fig.add_subplot(111); ax.axis('off')
                        ax.set_title(prefix.rsplit('.', 1)[0])
                        image = plt.imread(img)
                        ax.imshow(image)
                        ax.set_aspect('equal')
                        pdf.savefig(fig); plt.close(fig)
        results['pdf_outputs'] = pdf_path

    if make_pdf_filtered:
        pdf_f_path = os.path.join(output_dir, pdf_filtered_out or 'report_filtered_16610.pdf')
        with PdfPages(pdf_f_path) as pdf:
            # Portada filtrado
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.text(0.5, 0.7, 'Reporte filtrado ISO 16610 (selección)', ha='center', va='center', fontsize=18)
            fig.text(0.5, 0.6, f"Top {pdf_top_k} por grupo según {rank_metric}", ha='center', va='center', fontsize=10)
            pdf.savefig(fig); plt.close(fig)

            # Por grupo, tomar top-K según rank_metric y añadir imágenes 16610 del espécimen
            for gname in sorted(groups.keys()):
                # Reusar ranking ya calculado por grupo
                grows = groups[gname]
                def _rank_key(row, _metric=rank_metric):
                    val = _to_float(row.get(_metric))
                    return (val is None, val if val is not None else float('inf'))
                ranked = sorted(grows, key=_rank_key)[:max(1, pdf_top_k)]
                for r in ranked:
                    folder_rel = r.get('folder', '')
                    folder_abs = os.path.abspath(os.path.join(root_dir, folder_rel))
                    # Agregar perfil rugosidad 16610 y curva Rk 16610 si existen
                    for img_name, title in [
                        ('perfil_rugosidad_16610.png', f"{gname}/{r.get('specimen', '')} - Perfil 16610"),
                        ('curva_portancia_Rk_16610.png', f"{gname}/{r.get('specimen', '')} - Curva Rk 16610"),
                    ]:
                        img_path = os.path.join(folder_abs, img_name)
                        if os.path.exists(img_path):
                            fig = plt.figure(figsize=(11.69, 8.27))
                            ax = fig.add_subplot(111); ax.axis('off')
                            ax.set_title(title)
                            image = plt.imread(img_path)
                            ax.imshow(image)
                            ax.set_aspect('equal')
                            pdf.savefig(fig); plt.close(fig)
        results['pdf_filtered'] = pdf_f_path

    base_results = {
        'per_group_stats': per_group_stats,
        'groups_file': out_groups_csv,
        'report': out_md,
        'output_dir': output_dir,
    }
    base_results.update(results)
    return base_results


def main():
    """CLI para generar comparativas usando un summary existente o calculándolo en caliente."""
    parser = argparse.ArgumentParser(description='Comparación de métricas por grupo y entre grupos (a partir de batch_summary.json).')
    parser.add_argument('root', nargs='?', default=None, help='Carpeta raíz (por defecto ./data).')
    parser.add_argument('--summary', default='batch_summary.json', help='Ruta al batch_summary.json (relativa a root o absoluta).')
    parser.add_argument('--output-dir', default='reports', help='Carpeta de salida para reportes.')
    parser.add_argument('--metrics', nargs='*', default=['Ra', 'Rq', 'Rz_ISO', 'RSm', 'Rpk', 'Rk', 'Rvk', 'Mr1', 'Mr2'], help='Métricas a considerar.')
    parser.add_argument('--rank-metric', default='Ra', help='Métrica para ordenar rankings/plots (debe estar en --metrics).')
    parser.add_argument('--compute-if-missing', action='store_true', help='Calcular batch en caliente si no existe el summary.')
    parser.add_argument('--apply-filter', action='store_true', help='Si se calcula en caliente, aplicar filtrado ISO 16610.')
    parser.add_argument('--cutoff-mm', type=float, default=0.8, help='λc si se calcula en caliente.')
    parser.add_argument('--filter-source', choices=['primary', 'roughness'], default='primary', help='Fuente de filtrado si se calcula en caliente.')
    # PDF options
    parser.add_argument('--make-pdf', action='store_true', help='Generar PDF con salidas (rankings y comparativas).')
    parser.add_argument('--make-pdf-filtered', action='store_true', help='Generar PDF con imágenes filtradas ISO 16610 (selección top-K).')
    parser.add_argument('--pdf-out', default='report_outputs.pdf', help='Nombre del PDF de salidas.')
    parser.add_argument('--pdf-filtered-out', default='report_filtered_16610.pdf', help='Nombre del PDF filtrado.')
    parser.add_argument('--pdf-top-k', type=int, default=3, help='Top-K especímenes por grupo para incluir en PDF filtrado.')
    args = parser.parse_args()

    default_root = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'data'))
    root_dir = os.path.abspath(args.root) if args.root else default_root
    if not os.path.isdir(root_dir):
        print(f"Error: La carpeta raíz '{root_dir}' no existe.")
        sys.exit(1)

    summary_path = args.summary
    if not os.path.isabs(summary_path):
        summary_path = os.path.abspath(os.path.join(root_dir, summary_path))

    if not os.path.exists(summary_path):
        if not args.compute_if_missing:
            print("No se encontró el summary. Ejecute Batch.py o use --compute-if-missing.")
            sys.exit(2)
        if procesar_carpeta is None:
            print("No se puede calcular en caliente (procesar_carpeta no disponible).")
            sys.exit(3)
        # Calcular en caliente un resumen mínimo
        specimen_dirs = sorted(find_specimen_folders(root_dir))
        summary: List[Dict] = []
        failures: List[Dict] = []
        for folder in specimen_dirs:
            rel = os.path.relpath(folder, root_dir)
            try:
                result = procesar_carpeta(folder, apply_filter=args.apply_filter, cutoff_mm=args.cutoff_mm, filter_source=args.filter_source)
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
            except (FileNotFoundError, ValueError, OSError) as exc:
                failures.append({'folder': rel, 'error': str(exc)})
                print(f"  Error en '{rel}': {exc}")
        # Guardar JSON para reutilizar
        with open(summary_path, 'w', encoding='utf-8') as fh:
            json.dump({'summary': summary, 'failures': failures}, fh, ensure_ascii=False, indent=2)
        print(f"Summary calculado y guardado en: {summary_path}")

    # Salida
    out_dir = os.path.abspath(os.path.join(root_dir, args.output_dir))
    result = compare(
        root_dir, summary_path, out_dir, args.metrics, args.rank_metric,
        make_pdf=args.make_pdf, make_pdf_filtered=args.make_pdf_filtered,
        pdf_out=args.pdf_out, pdf_filtered_out=args.pdf_filtered_out, pdf_top_k=args.pdf_top_k,
    )
    print(f"Reporte generado en: {result['report']}")
    print(f"Comparación de grupos: {result['groups_file']}")
    if 'pdf_outputs' in result:
        print(f"PDF de salidas: {result['pdf_outputs']}")
    if 'pdf_filtered' in result:
        print(f"PDF (filtrado 16610): {result['pdf_filtered']}")


if __name__ == '__main__':
    main()
