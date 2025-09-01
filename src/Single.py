# -*- coding: utf-8 -*-
"""
Procesa un caso de medición de rugosidad desde una carpeta (p. ej. data/GrupoI/EspeI).
Calcula parámetros ISO 4287:1997 (incl. Rz ISO), familia Rk, exporta CSV y guarda gráficas.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import trapezoid
import csv
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
    with open(filepath, 'r', encoding='latin-1') as cfg_file:
        for line in cfg_file:
            parts = re.split(r'\t+', line.strip())
            if len(parts) >= 2:
                cfg_key = parts[0].strip()
                cfg_value = parts[1].strip()
                config_params[cfg_key] = cfg_value
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
    # utf-8-sig agrega BOM para que Excel muestre acentos correctamente
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as out_fh:
        out_writer = csv.writer(out_fh)
        out_writer.writerow(['Parámetro', 'Valor'])
        for pname, pvalue in resultados_dict.items():
            out_writer.writerow([pname, pvalue])

def leer_perfil_con_header(filepath):
    """Lee un perfil Surfcom con cabecera: retorna (longitud_mm, n_puntos, perfil ndarray)."""
    if not os.path.exists(filepath):
        print(f"Error: El archivo de datos '{filepath}' no fue encontrado.")
        return None
    try:
        with open(filepath, 'r', encoding='latin-1') as fh_in:
            # Línea 1: longitud (mm), línea 2: número de puntos
            longitud_mm = float(next(fh_in).strip())
            n_puntos = int(next(fh_in).strip())
            datos = []
            for line in fh_in:
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
        with open(filepath, 'r', encoding='utf-8') as fh_in:
            # Omitir las 2 líneas de cabecera
            next(fh_in)
            next(fh_in)
            data = []
            for line in fh_in:
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
    ISO 13565-2: cálculo de Rk, Rpk, Rvk, Mr1, Mr2 desde la curva Abbott–Firestone.

    Método:
      1) Construye la curva de razón de material (MRC) como altura vs r (%)
      2) Ajuste lineal al tramo más lineal (ventana deslizante de 40 %)
      3) Intersecciones r1 y r2 donde MRC = línea del núcleo
      4) Áreas A1 (0..r1) y A2 (r2..100) para Rpk y Rvk
         Rpk = A1 / r1 ; Rvk = A2 / (100 - r2)
         Rk = altura_línea(r1) - altura_línea(r2)
    """
    if perfil is None or len(perfil) < 10:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # 1) MRC: ordenar alturas (desc) y asignar r en 0..100 %
    z = np.sort(perfil)[::-1]
    n = len(z)
    r = np.linspace(0.0, 100.0, n)

    # 2) Buscar tramo más lineal con ventana ~40% del dominio
    win_pct = 40.0
    win_len = max(5, int(n * (win_pct / 100.0)))
    _best_i, best_rmse = 0, np.inf
    best_coef = (0.0, float(z.mean()))
    for i in range(0, n - win_len):
        rr = r[i:i + win_len]
        zz = z[i:i + win_len]
        a, b = np.polyfit(rr, zz, 1)  # z ≈ a*r + b
        zz_hat = a * rr + b
        rmse = float(np.sqrt(np.mean((zz - zz_hat) ** 2)))
        if rmse < best_rmse:
            best_rmse, _best_i, best_coef = rmse, i, (a, b)

    a, b = best_coef

    # 3) Intersecciones r1 y r2 (MRC - línea = 0)
    delta = z - (a * r + b)
    # Buscar primeros y últimos cruces de signo
    sign = np.sign(delta)
    crossings = np.where(np.diff(sign))[0]
    if crossings.size >= 2:
        i1 = crossings[0]
        i2 = crossings[-1]
        # Interpolación lineal para r1, r2
        def interp_r(i):
            r0, r1_ = r[i], r[i + 1]
            d0, d1 = delta[i], delta[i + 1]
            t = 0.0 if (d1 - d0) == 0 else (-d0) / (d1 - d0)
            return r0 + t * (r1_ - r0)
        r1, r2 = float(interp_r(i1)), float(interp_r(i2))
    elif crossings.size == 1:
        i1 = crossings[0]
        r1 = float(r[i1])
        r2 = 100.0
    else:
        r1, r2 = 0.0, 100.0

    # 4) Áreas A1 y A2 por trapecios
    # Construir funciones discretas en rejilla uniforme r
    z_line = a * r + b
    # A1: entre 0..r1 de (z - z_line) positivo
    mask1 = r <= r1
    A1 = float(trapezoid(np.maximum(z[mask1] - z_line[mask1], 0.0), r[mask1])) if np.any(mask1) else 0.0
    # A2: entre r2..100 de (z_line - z) positivo
    mask2 = r >= r2
    A2 = float(trapezoid(np.maximum(z_line[mask2] - z[mask2], 0.0), r[mask2])) if np.any(mask2) else 0.0

    Mr1 = float(r1)
    Mr2 = float(r2)
    Rpk = float(A1 / r1) if r1 > 0 else 0.0
    Rvk = float(A2 / (100.0 - r2)) if r2 < 100.0 else 0.0
    Rk = float(abs((a * r1 + b) - (a * r2 + b)))

    if plot:
        plt.figure(figsize=(10, 7))
        plt.plot(r, z, color='black', label='Curva de Portancia (MRC)')
        plt.plot(r, z_line, 'r--', label='Línea del Núcleo (ISO 13565-2)')
        # Marcas r1, r2
        z1, z2 = a * r1 + b, a * r2 + b
        plt.scatter([r1, r2], [z1, z2], c=['b', 'g'], zorder=3)
        plt.vlines([r1, r2], [min(z.min(), z_line.min())]*2, [z1, z2], colors=['b', 'g'], linestyles=':')
        # Anotaciones
        plt.text(r1 / 2.0, z1 + 0.05 * abs(z1), f'Rpk = {Rpk:.2f} µm', color='b', ha='center')
        plt.text(r1 + (r2 - r1) / 2.0, (z1 + z2) / 2.0, f'Rk = {Rk:.2f} µm', color='r', ha='center')
        plt.text((100 + r2) / 2.0, z2 - 0.05 * abs(z2), f'Rvk = {Rvk:.2f} µm', color='g', ha='center')
        plt.title('Curva de Material Portante y parámetros ISO 13565-2')
        plt.xlabel('Material portante (%)')
        plt.ylabel('Altura del Perfil (µm)')
        plt.grid(True)
        plt.legend()
        plt.savefig(filename)
        print(f"Gráfica '{filename}' guardada correctamente.")

    return Rk, Rpk, Rvk, Mr1, Mr2


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
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as out_fh:
        out_writer = csv.writer(out_fh)
        out_writer.writerow(['x (mm)', 'altura (µm)'])
        for x, y in zip(eje_x, perfil):
            out_writer.writerow([f"{x:.9f}", f"{y:.9f}"])


# =============================================================================
# --- EJECUCIÓN DEL ANÁLISIS ---
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Análisis de rugosidad (ISO 4287:1997 e ISO 13565-2) con opción de filtrado ISO 16610.')
    parser.add_argument('folder', nargs='?', default=None, help='Carpeta con 3.tx1, 3.tx2, 3.tx3 (por defecto data/GrupoI/EspeI).')
    parser.add_argument('--apply-filter', action='store_true', help='Aplicar filtrado ISO 16610 a partir del perfil primario para obtener rugosidad.')
    parser.add_argument('--cutoff-mm', type=float, default=0.8, help='Longitud de corte (λc) en mm para el filtro Gaussiano (p.ej., 0.8, 2.5).')
    parser.add_argument('--filter-source', choices=['primary', 'roughness'], default='primary', help='Fuente para el filtrado (primary recomendado).')
    args = parser.parse_args()

    # Carpeta a procesar: se puede pasar por argumento; por defecto: data/GrupoI/EspeI
    default_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'GrupoI', 'EspeI'))
    base_dir = os.path.abspath(args.folder) if args.folder else default_dir

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

    # --- Cálculos (perfil de rugosidad original) ---
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

    # --- Filtrado ISO 16610 opcional ---
    if args.apply_filter:
        # Relación sigma-samples para -3 dB en λc: sigma = 0.1325 * λc; sigma_samples = sigma / dx
        def compute_sigma_samples(x_mm, lambda_c_mm):
            dx = float(x_mm[1] - x_mm[0]) if len(x_mm) > 1 else lambda_c_mm
            sigma_mm = 0.1325 * float(lambda_c_mm)
            return max(0.5, sigma_mm / dx)

        if args.filter_source == 'primary':
            sigma_samp = compute_sigma_samples(eje_x_primario, args.cutoff_mm)
            waviness = gaussian_filter1d(perfil_primario, sigma=sigma_samp, mode='nearest')
            rug_16610 = perfil_primario - waviness
            x_16610 = eje_x_primario
        else:
            sigma_samp = compute_sigma_samples(eje_x_rug, args.cutoff_mm)
            waviness = gaussian_filter1d(perfil_rugosidad, sigma=sigma_samp, mode='nearest')
            rug_16610 = perfil_rugosidad - waviness
            x_16610 = eje_x_rug

        # Parámetros sobre rug_16610
        Ra_f = float(np.mean(np.abs(rug_16610)))
        Rq_f = float(np.sqrt(np.mean(rug_16610**2)))
        Rp_f = float(np.max(rug_16610))
        Rv_f = float(np.min(rug_16610))
        Rz_Rt_f = float(Rp_f - Rv_f)
        Rsk_f = float(stats.skew(rug_16610))
        Rku_f = float(stats.kurtosis(rug_16610, fisher=False))
        RSm_f = float(calcular_rsm(rug_16610, x_16610))
        Rk_f, Rpk_f, Rvk_f, Mr1_f, Mr2_f = calcular_parametros_rk(
            rug_16610, plot=True, filename=os.path.join(base_dir, 'curva_portancia_Rk_16610.png')
        )
        Rz_ISO_f = float(calcular_rz_iso(rug_16610))

        # CSV extendido con sufijo ISO 16610
        resultados_16610 = {
            'Ra (ISO 16610)': f'{Ra_f:.3f} µm',
            'Rq (ISO 16610)': f'{Rq_f:.3f} µm',
            'Rp (ISO 16610)': f'{Rp_f:.3f} µm',
            'Rv (ISO 16610)': f'{Rv_f:.3f} µm',
            'Rz/Rt (ISO 16610)': f'{Rz_Rt_f:.3f} µm',
            'Rz (ISO 4287:1997, perfil 16610)': f'{Rz_ISO_f:.3f} µm',
            'Rsk (ISO 16610)': f'{Rsk_f:.3f}',
            'Rku (ISO 16610)': f'{Rku_f:.3f}',
            'RSm (ISO 16610)': f'{RSm_f:.3f} µm',
            'Rpk (ISO 13565-2, 16610)': f'{Rpk_f:.3f} µm',
            'Rk (ISO 13565-2, 16610)': f'{Rk_f:.3f} µm',
            'Rvk (ISO 13565-2, 16610)': f'{Rvk_f:.3f} µm',
            'Mr1 (ISO 13565-2, 16610) (%)': f'{Mr1_f:.2f}',
            'Mr2 (ISO 13565-2, 16610) (%)': f'{Mr2_f:.2f}',
            'λc (mm)': f'{args.cutoff_mm:.3f}',
            'Fuente filtrado': args.filter_source,
        }

        # Append al CSV principal
        try:
            with open(csv_path, 'a', newline='', encoding='utf-8-sig') as append_fh:
                append_writer = csv.writer(append_fh)
                append_writer.writerow([])
                append_writer.writerow(['Parámetro (ISO 16610/13565-2)', 'Valor'])
                for pname, pvalue in resultados_16610.items():
                    append_writer.writerow([pname, pvalue])
            print(f"Resultados ISO 16610 añadidos a '{csv_path}'.")
        except (OSError, csv.Error) as e:
            print(f"No se pudieron anexar resultados ISO 16610: {e}")

        # Gráfica y CSV del perfil 16610
        plt.figure(figsize=(12, 6)); plt.plot(x_16610, rug_16610, color='purple', lw=1)
        plt.title(f'Perfil de Rugosidad (ISO 16610, λc={args.cutoff_mm} mm)')
        plt.xlabel('Distancia (mm)'); plt.ylabel('Altura (µm)')
        plt.grid(True); plt.savefig(os.path.join(base_dir, 'perfil_rugosidad_16610.png'))
        print(f"Gráfica '{os.path.join(base_dir, 'perfil_rugosidad_16610.png')}' guardada correctamente.")
        exportar_perfil_csv(x_16610, rug_16610, os.path.join(base_dir, 'perfil_rugosidad_16610.csv'))
