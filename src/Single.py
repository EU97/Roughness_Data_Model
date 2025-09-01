# -*- coding: utf-8 -*-
"""
Procesa un caso de medición de rugosidad desde una carpeta (p. ej. data/GrupoI/EspeI).
Calcula parámetros ISO 4287:1997 (incl. Rz ISO), familia Rk, exporta CSV y guarda gráficas.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import re
import os
import sys

# Mejorar salida para acentos/µ: forzar UTF-8 en stdout si es posible
try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError, OSError):
    pass


def parse_config_file(filepath):
    """
    Lee un archivo de configuración (.tx3) y extrae los parámetros clave.
    Utiliza codificación 'latin-1' para compatibilidad con software de medición.
    """
    if not os.path.exists(filepath):
        print(f"Error: El archivo de configuración '{filepath}' no fue encontrado.")
        return None
        
    config_params = {}
    with open(filepath, 'r', encoding='latin-1') as f:
        for line in f:
            parts = re.split(r'\t+', line.strip())
            if len(parts) >= 2:
                key = parts[0].strip()
                value = parts[1].strip()
                config_params[key] = value
    return config_params

def calcular_rz_iso(perfil: np.ndarray, segmentos: int = 5) -> float:
    """Calcula Rz según ISO 4287:1997.

    Aproximación práctica: dividir el perfil en 5 segmentos iguales (5 longitudes de muestreo)
    y promediar la diferencia pico-valle máxima de cada segmento.
    """
    if perfil is None or len(perfil) == 0:
        return 0.0
    n = len(perfil)
    seg_len = n // segmentos
    if seg_len == 0:
        return float(np.max(perfil) - np.min(perfil))
    rz_vals = []
    for i in range(segmentos):
        ini, fin = i * seg_len, (i + 1) * seg_len if i < segmentos - 1 else n
        seg = perfil[ini:fin]
        if seg.size:
            rz_vals.append(np.max(seg) - np.min(seg))
    return float(np.mean(rz_vals)) if rz_vals else 0.0

def exportar_resultados_csv(resultados_dict: dict, filepath: str) -> None:
    """Exporta los resultados de rugosidad a un archivo CSV en filepath."""
    import csv
    # utf-8-sig agrega BOM para que Excel muestre acentos correctamente
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Parámetro', 'Valor'])
        for k, v in resultados_dict.items():
            writer.writerow([k, v])

def leer_perfil_con_header(filepath):
    """Lee un perfil Surfcom con cabecera: retorna (longitud_mm, n_puntos, perfil ndarray)."""
    if not os.path.exists(filepath):
        print(f"Error: El archivo de datos '{filepath}' no fue encontrado.")
        return None
    try:
        with open(filepath, 'r', encoding='latin-1') as f:
            # Línea 1: longitud (mm), línea 2: número de puntos
            longitud_mm = float(next(f).strip())
            n_puntos = int(next(f).strip())
            datos = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                datos.append(float(line))
        perfil = np.array(datos, dtype=float)
        # Si el archivo trae más/menos puntos de los esperados, recortar/avisar
        if len(perfil) != n_puntos:
            print(f"Aviso: Se esperaban {n_puntos} puntos pero se leyeron {len(perfil)} en '{os.path.basename(filepath)}'.")
        return longitud_mm, n_puntos, perfil
    except (ValueError, OSError) as e:
        print(f"Error leyendo '{filepath}': {e}")
        return None

def leer_perfil_txt(filepath):
    """
    Lee un archivo de perfil de rugosidad (.tx1, .tx2).
    """
    if not os.path.exists(filepath):
        print(f"Error: El archivo de datos '{filepath}' no fue encontrado.")
        return None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Omitir las 2 líneas de cabecera
            next(f)
            next(f)
            data = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(float(line))
        return np.array(data, dtype=float)
    except ValueError:
        print(f"Error: El archivo '{filepath}' contiene datos no numéricos.")
        return None

def calcular_rsm(perfil, eje_x):
    """Calcula RSm (Anchura Media de los Elementos del Perfil) en µm."""
    zero_crossings = np.where(np.diff(np.sign(perfil)))[0]
    if len(zero_crossings) < 2: return 0.0
    
    element_lengths_pts = np.diff(zero_crossings)
    rango = (eje_x[-1] - eje_x[0])
    if rango <= 0:
        return 0.0
    pts_por_mm = len(perfil) / rango
    element_lengths_mm = element_lengths_pts / pts_por_mm
    return np.mean(element_lengths_mm) * 1000

def calcular_parametros_rk(perfil, plot=False, filename='curva_portancia_Rk.png'):
    """
    Estima Rk, Rpk, Rvk, Mr1 y Mr2 desde la curva Abbott–Firestone.
    Nota: Heurístico; para resultados normalizados, implementar ISO 13565-2.
    """
    if perfil is None or len(perfil) < 10:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    n_bins = min(2048, max(256, len(perfil) // 2))
    hist, edges = np.histogram(perfil, bins=n_bins, density=True)
    heights = (edges[:-1] + edges[1:]) / 2.0
    step = float(heights[1] - heights[0]) if len(heights) > 1 else 1.0

    # Material ratio curve (descending cumulative)
    mrc = np.cumsum(hist[::-1]) * step
    mrc_heights = heights[::-1]
    total = mrc[-1] if mrc[-1] != 0 else 1.0
    mrc_percent = (mrc / total) * 100.0

    window_size = max(10, int(0.4 * len(mrc_percent)))
    if window_size >= len(mrc_percent):
        window_size = max(2, len(mrc_percent) // 3)

    best_i = 0
    best_slope = np.inf
    for i in range(0, len(mrc_percent) - window_size):
        dy = mrc_percent[i + window_size] - mrc_percent[i]
        dx = mrc_heights[i + window_size] - mrc_heights[i]
        if dx == 0:
            continue
        slope = abs(dy / dx)
        if 0 < slope < best_slope:
            best_slope = slope
            best_i = i

    i0, j0 = best_i, best_i + window_size
    m_line = (mrc_percent[j0] - mrc_percent[i0]) / (mrc_heights[j0] - mrc_heights[i0])
    c_line = mrc_percent[i0] - m_line * mrc_heights[i0]

    h_mr1 = (0.0 - c_line) / m_line
    h_mr2 = (100.0 - c_line) / m_line

    rk_core_v = abs(h_mr1 - h_mr2)
    rpk_peaks_v = float(np.max(perfil) - h_mr1)
    rvk_valleys_v = float(h_mr2 - np.min(perfil))
    mr1_ratio_v = float(np.interp(h_mr1, mrc_heights[::-1], mrc_percent[::-1]))
    mr2_ratio_v = float(np.interp(h_mr2, mrc_heights[::-1], mrc_percent[::-1]))

    if plot:
        plt.figure(figsize=(10, 7))
        plt.plot(mrc_percent, mrc_heights, color='black', label='Curva de Portancia')
        plt.plot([mr1_ratio_v, mr2_ratio_v], [h_mr1, h_mr2], 'r--', label='Línea del Núcleo')
        plt.hlines(y=h_mr1, xmin=0, xmax=mr1_ratio_v, color='b', ls=':')
        plt.text(mr1_ratio_v / 2.0, h_mr1 + 1.0, f'Rpk = {rpk_peaks_v:.2f} µm', c='b', ha='center')
        plt.vlines(x=mr1_ratio_v, ymin=h_mr2, ymax=h_mr1, color='r', ls='-')
        plt.text(mr1_ratio_v + 5.0, (h_mr1 + h_mr2) / 2.0, f'Rk = {rk_core_v:.2f} µm', c='r')
        plt.hlines(y=h_mr2, xmin=mr2_ratio_v, xmax=100, color='g', ls=':')
        plt.text((100 + mr2_ratio_v) / 2.0, h_mr2 - 1.0, f'Rvk = {rvk_valleys_v:.2f} µm', c='g', ha='center')
        plt.title('Curva de Material Portante (Abbott-Firestone) y Análisis Rk')
        plt.xlabel('Ratio de Material Portante (%)')
        plt.ylabel('Altura del Perfil (µm)')
        plt.grid(True)
        plt.legend()
        plt.savefig(filename)
        print(f"Gráfica '{filename}' guardada correctamente.")

    return rk_core_v, rpk_peaks_v, rvk_valleys_v, mr1_ratio_v, mr2_ratio_v


def corregir_pendiente(perfil: np.ndarray, eje_x: np.ndarray):
    """Elimina la pendiente lineal (tilt) del perfil usando ajuste por mínimos cuadrados.
    Devuelve (perfil_corregido, pendiente, intercepto).
    """
    if perfil.size != eje_x.size or perfil.size == 0:
        return perfil, 0.0, 0.0
    # Ajuste lineal y = m x + b
    m, b = np.polyfit(eje_x, perfil, 1)
    tendencia = m * eje_x + b
    perfil_corr = perfil - tendencia
    return perfil_corr, float(m), float(b)


def exportar_perfil_csv(eje_x: np.ndarray, perfil: np.ndarray, filepath: str):
    """Exporta un perfil (x, y) a CSV con BOM para Excel."""
    import csv
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['x (mm)', 'altura (µm)'])
        for x, y in zip(eje_x, perfil):
            w.writerow([f"{x:.9f}", f"{y:.9f}"])


# =============================================================================
# --- EJECUCIÓN DEL ANÁLISIS ---
# =============================================================================
if __name__ == "__main__":

    # Carpeta a procesar: se puede pasar por argumento; por defecto: data/GrupoI/EspeI
    default_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'GrupoI', 'EspeI'))
    base_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else default_dir

    if not os.path.isdir(base_dir):
        print(f"Error: La carpeta '{base_dir}' no existe.")
        sys.exit(1)

    ruta_archivo_config = os.path.join(base_dir, '3.tx3')
    ruta_archivo_primario = os.path.join(base_dir, '3.tx1')
    ruta_archivo_rugosidad = os.path.join(base_dir, '3.tx2')

    params = parse_config_file(ruta_archivo_config)

    # Leer perfiles con cabecera (longitud, puntos, datos)
    header_primario = leer_perfil_con_header(ruta_archivo_primario)
    header_rugosidad = leer_perfil_con_header(ruta_archivo_rugosidad)

    if not header_primario or not header_rugosidad:
        print("Error: No fue posible leer los perfiles primario y/o de rugosidad.")
        sys.exit(1)

    long_mm_primario, _, perfil_primario = header_primario
    long_mm_rug, _, perfil_rugosidad = header_rugosidad

    # Construir ejes X independientes según longitudes declaradas
    eje_x_primario = np.linspace(0, float(long_mm_primario), len(perfil_primario))
    eje_x_rug = np.linspace(0, float(long_mm_rug), len(perfil_rugosidad))

    # --- Cálculos ---
    Ra = float(np.mean(np.abs(perfil_rugosidad)))
    Rq = float(np.sqrt(np.mean(perfil_rugosidad**2)))
    Rp = float(np.max(perfil_rugosidad))
    Rv = float(np.min(perfil_rugosidad))
    Rz_Rt = float(Rp - Rv)
    Rsk = float(stats.skew(perfil_rugosidad))
    Rku = float(stats.kurtosis(perfil_rugosidad, fisher=False))
    RSm = float(calcular_rsm(perfil_rugosidad, eje_x_rug))
    rk_core, rpk_peaks, rvk_valleys, mr1_ratio, mr2_ratio = calcular_parametros_rk(
        perfil_rugosidad, plot=True, filename=os.path.join(base_dir, 'curva_portancia_Rk.png')
    )
    Rz_ISO = float(calcular_rz_iso(perfil_rugosidad))

    # --- Exportar resultados ---
    resultados = {
        'Ra (Promedio Aritmético)': f'{Ra:.3f} µm',
        'Rq (RMS)': f'{Rq:.3f} µm',
        'Rp (Pico Máximo)': f'{Rp:.3f} µm',
        'Rv (Valle Máximo)': f'{Rv:.3f} µm',
        'Rz/Rt (Altura Total)': f'{Rz_Rt:.3f} µm',
        'Rz (ISO 4287:1997)': f'{Rz_ISO:.3f} µm',
        'Rsk (Asimetría)': f'{Rsk:.3f}',
        'Rku (Curtosis)': f'{Rku:.3f}',
        'RSm (Anchura Media de Elementos)': f'{RSm:.3f} µm',
    'Rpk (Altura de picos que se desgastan)': f'{rpk_peaks:.3f} µm',
    'Rk (Profundidad del núcleo funcional)': f'{rk_core:.3f} µm',
    'Rvk (Profundidad de valles para lubricante)': f'{rvk_valleys:.3f} µm',
    'Mr1 (%)': f'{mr1_ratio:.2f}',
    'Mr2 (%)': f'{mr2_ratio:.2f}'
    }
    csv_path = os.path.join(base_dir, 'resultados_rugosidad.csv')
    exportar_resultados_csv(resultados, csv_path)
    print(f"\nResultados exportados a '{csv_path}'.")

    # --- Impresión de Resultados ---
    print("\n--- ANÁLISIS COMPLETO DE RUGOSIDAD ---")
    print("\nParámetros de Amplitud:")
    print(f"  Ra: {Ra:.3f} µm\t(Promedio Aritmético)")
    print(f"  Rq: {Rq:.3f} µm\t(RMS)")
    print(f"  Rp: {Rp:.3f} µm\t(Pico Máximo)")
    print(f"  Rv: {Rv:.3f} µm\t(Valle Máximo)")
    print(f"  Rz/Rt: {Rz_Rt:.3f} µm\t(Altura Total)")
    print(f"  Rz (ISO 4287:1997): {Rz_ISO:.3f} µm\t(Media de 5 segmentos)")
    print(f"  Rsk: {Rsk:.3f}\t(Asimetría)")
    print(f"  Rku: {Rku:.3f}\t(Curtosis)")
    print("\nParámetros de Espaciado:")
    print(f"  RSm: {RSm:.3f} µm\t(Anchura Media de Elementos)")
    print("\nParámetros Funcionales (Familia Rk):")
    print(f"  Rpk: {rpk_peaks:.3f} µm\t(Altura de picos que se desgastan)")
    print(f"  Rk: {rk_core:.3f} µm\t(Profundidad del núcleo funcional)")
    print(f"  Rvk: {rvk_valleys:.3f} µm\t(Profundidad de valles para lubricante)")
    print(f"  Mr1: {mr1_ratio:.2f} %\t(Material ratio inferior)")
    print(f"  Mr2: {mr2_ratio:.2f} %\t(Material ratio superior)")

    # --- Generación de Gráficas de Perfil ---
    plt.figure(figsize=(12, 6)); plt.plot(eje_x_primario, perfil_primario, color='black', lw=1)
    plt.title('Perfil Primario'); plt.xlabel('Distancia (mm)'); plt.ylabel('Altura (µm)')
    plt.grid(True); plt.savefig(os.path.join(base_dir, 'perfil_primario.png'))
    print(f"Gráfica '{os.path.join(base_dir, 'perfil_primario.png')}' guardada correctamente.")

    plt.figure(figsize=(12, 6)); plt.plot(eje_x_rug, perfil_rugosidad, color='blue', lw=1)
    plt.title('Perfil de Rugosidad'); plt.xlabel('Distancia (mm)'); plt.ylabel('Altura (µm)')
    plt.grid(True); plt.savefig(os.path.join(base_dir, 'perfil_rugosidad.png'))
    print(f"Gráfica '{os.path.join(base_dir, 'perfil_rugosidad.png')}' guardada correctamente.")

    # --- Perfiles con corrección de pendiente (sin alterar salidas previas) ---
    prim_corr, m1, b1 = corregir_pendiente(perfil_primario, eje_x_primario)
    rug_corr, m2, b2 = corregir_pendiente(perfil_rugosidad, eje_x_rug)

    plt.figure(figsize=(12, 6)); plt.plot(eje_x_primario, prim_corr, color='black', lw=1)
    plt.title('Perfil Primario (corrección de pendiente)')
    plt.xlabel('Distancia (mm)'); plt.ylabel('Altura (µm)')
    plt.grid(True); plt.savefig(os.path.join(base_dir, 'perfil_primario_corr.png'))
    print(f"Gráfica '{os.path.join(base_dir, 'perfil_primario_corr.png')}' guardada correctamente.")

    plt.figure(figsize=(12, 6)); plt.plot(eje_x_rug, rug_corr, color='blue', lw=1)
    plt.title('Perfil de Rugosidad (corrección de pendiente)')
    plt.xlabel('Distancia (mm)'); plt.ylabel('Altura (µm)')
    plt.grid(True); plt.savefig(os.path.join(base_dir, 'perfil_rugosidad_corr.png'))
    print(f"Gráfica '{os.path.join(base_dir, 'perfil_rugosidad_corr.png')}' guardada correctamente.")

    # CSV opcionales de perfiles corregidos
    exportar_perfil_csv(eje_x_primario, prim_corr, os.path.join(base_dir, 'perfil_primario_corr.csv'))
    exportar_perfil_csv(eje_x_rug, rug_corr, os.path.join(base_dir, 'perfil_rugosidad_corr.csv'))

    print("\nAnálisis finalizado.")
