import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import re
import os

# =============================================================================
# --- CONFIGURACIÓN DE ARCHIVOS ---
# Modifica estas rutas para que apunten a la ubicación de tus archivos.
# Si los archivos están en la misma carpeta que el script, solo necesitas el nombre.
# =============================================================================

# Obtener el directorio donde está el script
script_dir = os.path.dirname(os.path.abspath(__file__))
ruta_archivo_config = os.path.join(script_dir, '3.tx3')
ruta_archivo_primario = os.path.join(script_dir, '3.tx1')
ruta_archivo_rugosidad = os.path.join(script_dir, '3.tx2')


# =============================================================================
# --- FUNCIONES DE ANÁLISIS ---
# =============================================================================

def calcular_rz_iso(perfil, segmentos=5):
    """
    Calcula Rz según ISO 4287: media de las cinco mayores diferencias pico-valle en cinco segmentos iguales.
    """
    n = len(perfil)
    seg_len = n // segmentos
    rz_vals = []
    for i in range(segmentos):
        seg = perfil[i*seg_len:(i+1)*seg_len]
        if len(seg) > 0:
            rz_vals.append(np.max(seg) - np.min(seg))
    return np.mean(rz_vals) if rz_vals else 0.0

def exportar_resultados_csv(resultados, filename='resultados_rugosidad.csv'):
    """
    Exporta los resultados de rugosidad a un archivo CSV.
    """
    import csv
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Parametro', 'Valor'])
        for k, v in resultados.items():
            writer.writerow([k, v])
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

def leer_perfil_txt(filepath):
    """
    Lee un archivo de perfil de rugosidad (.tx1, .tx2).
    """
    if not os.path.exists(filepath):
        print(f"Error: El archivo de datos '{filepath}' no fue encontrado.")
        return None
        
    try:
        with open(filepath, 'r') as f:
            next(f); next(f) # Omitir las 2 líneas de cabecera
            data = [float(line.strip()) for line in f]
        return np.array(data)
    except ValueError:
        print(f"Error: El archivo '{filepath}' contiene datos no numéricos.")
        return None

def calcular_rsm(perfil, eje_x):
    """Calcula RSm (Anchura Media de los Elementos del Perfil) en µm."""
    zero_crossings = np.where(np.diff(np.sign(perfil)))[0]
    if len(zero_crossings) < 2: return 0.0
    
    element_lengths_pts = np.diff(zero_crossings)
    pts_por_mm = len(perfil) / (eje_x[-1] - eje_x[0])
    element_lengths_mm = element_lengths_pts / pts_por_mm
    return np.mean(element_lengths_mm) * 1000

def calcular_parametros_rk(perfil, plot=False, filename='curva_portancia_Rk.png'):
    """
    Calcula los parámetros de la familia Rk (Rk, Rpk, Rvk) a partir de la
    curva de material portante (Abbott-Firestone).
    """
    n_bins = 2048
    hist, bin_edges = np.histogram(perfil, bins=n_bins, density=True)
    heights = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    mrc = np.cumsum(hist[::-1]) * np.diff(heights)[0]
    mrc_heights = heights[::-1]
    mrc_percent = mrc / mrc[-1] * 100

    window_size = int(0.4 * len(mrc_percent))
    min_slope_idx = -1
    min_slope = np.inf

    for i in range(len(mrc_percent) - window_size):
        dy = mrc_percent[i+window_size] - mrc_percent[i]
        dx = mrc_heights[i+window_size] - mrc_heights[i]
        if dx == 0: continue
        slope = abs(dy / dx)
        if 0 < slope < min_slope:
            min_slope = slope
            min_slope_idx = i

    i, j = min_slope_idx, min_slope_idx + window_size
    m = (mrc_percent[j] - mrc_percent[i]) / (mrc_heights[j] - mrc_heights[i])
    c = mrc_percent[i] - m * mrc_heights[i]
    
    h_mr1, h_mr2 = (0 - c) / m, (100 - c) / m
    Rk, Rpk, Rvk = abs(h_mr1 - h_mr2), np.max(perfil) - h_mr1, h_mr2 - np.min(perfil)
    Mr1 = np.interp(h_mr1, mrc_heights[::-1], mrc_percent[::-1])
    Mr2 = np.interp(h_mr2, mrc_heights[::-1], mrc_percent[::-1])
    
    if plot:
        plt.figure(figsize=(10, 7))
        plt.plot(mrc_percent, mrc_heights, color='black', label='Curva de Portancia')
        plt.plot([Mr1, Mr2], [h_mr1, h_mr2], 'r--', label='Línea del Núcleo')
        plt.hlines(y=h_mr1, xmin=0, xmax=Mr1, color='b', ls=':')
        plt.text(Mr1/2, h_mr1 + 1, f'Rpk = {Rpk:.2f} µm', c='b', ha='center')
        plt.vlines(x=Mr1, ymin=h_mr2, ymax=h_mr1, color='r', ls='-')
        plt.text(Mr1 + 5, (h_mr1 + h_mr2)/2, f'Rk = {Rk:.2f} µm', c='r')
        plt.hlines(y=h_mr2, xmin=Mr2, xmax=100, color='g', ls=':')
        plt.text((100+Mr2)/2, h_mr2 - 1, f'Rvk = {Rvk:.2f} µm', c='g', ha='center')
        plt.title('Curva de Material Portante (Abbott-Firestone) y Análisis Rk')
        plt.xlabel('Ratio de Material Portante (%)'); plt.ylabel('Altura del Perfil (µm)')
        plt.grid(True); plt.legend()
        plt.savefig(filename)
        print(f"Gráfica '{filename}' guardada correctamente.")

    return Rk, Rpk, Rvk, Mr1, Mr2


# =============================================================================
# --- EJECUCIÓN DEL ANÁLISIS ---
# =============================================================================
if __name__ == "__main__":
    
    params = parse_config_file(ruta_archivo_config)
    
    if params:
        perfil_primario = leer_perfil_txt(ruta_archivo_primario)
        perfil_rugosidad = leer_perfil_txt(ruta_archivo_rugosidad)

        if perfil_primario is not None and perfil_rugosidad is not None:
            # --- Preparación ---
            longitud_medicion_mm = float(params.get('Longitud medición', '0.0mm').replace('mm', ''))
            eje_x = np.linspace(0, longitud_medicion_mm, len(perfil_primario))
            
            # --- Cálculos ---
            Ra = np.mean(np.abs(perfil_rugosidad)); Rq = np.sqrt(np.mean(perfil_rugosidad**2))
            Rp = np.max(perfil_rugosidad); Rv = np.min(perfil_rugosidad); Rz_Rt = Rp - Rv
            Rsk = stats.skew(perfil_rugosidad); Rku = stats.kurtosis(perfil_rugosidad, fisher=False)
            RSm = calcular_rsm(perfil_rugosidad, eje_x)
            Rk, Rpk, Rvk, Mr1, Mr2 = calcular_parametros_rk(perfil_rugosidad, plot=True)
            Rz_ISO = calcular_rz_iso(perfil_rugosidad)
            
            # --- Exportar resultados ---
            resultados = {
                'Ra (Promedio Aritmético)': f'{Ra:.3f} µm',
                'Rq (RMS)': f'{Rq:.3f} µm',
                'Rp (Pico Máximo)': f'{Rp:.3f} µm',
                'Rv (Valle Máximo)': f'{Rv:.3f} µm',
                'Rz/Rt (Altura Total)': f'{Rz_Rt:.3f} µm',
                'Rz (ISO 4287)': f'{Rz_ISO:.3f} µm',
                'Rsk (Asimetría)': f'{Rsk:.3f}',
                'Rku (Curtosis)': f'{Rku:.3f}',
                'RSm (Anchura Media de Elementos)': f'{RSm:.3f} µm',
                'Rpk (Altura de picos que se desgastan)': f'{Rpk:.3f} µm',
                'Rk (Profundidad del núcleo funcional)': f'{Rk:.3f} µm',
                'Rvk (Profundidad de valles para lubricante)': f'{Rvk:.3f} µm',
                'Mr1 (%)': f'{Mr1:.2f}',
                'Mr2 (%)': f'{Mr2:.2f}'
            }
            exportar_resultados_csv(resultados)
            print("\nResultados exportados a 'resultados_rugosidad.csv'.")
            # --- Impresión de Resultados ---
            print("\n--- ANÁLISIS COMPLETO DE RUGOSIDAD ---")
            print("\nParámetros de Amplitud:")
            print(f"  Ra: {Ra:.3f} µm\t(Promedio Aritmético)")
            print(f"  Rq: {Rq:.3f} µm\t(RMS)")
            print(f"  Rp: {Rp:.3f} µm\t(Pico Máximo)")
            print(f"  Rv: {Rv:.3f} µm\t(Valle Máximo)")
            print(f"  Rz/Rt: {Rz_Rt:.3f} µm\t(Altura Total)")
            print(f"  Rz (ISO 4287): {Rz_ISO:.3f} µm\t(Media de 5 segmentos)")
            print(f"  Rsk: {Rsk:.3f}\t(Asimetría)")
            print(f"  Rku: {Rku:.3f}\t(Curtosis)")
            print("\nParámetros de Espaciado:")
            print(f"  RSm: {RSm:.3f} µm\t(Anchura Media de Elementos)")
            print("\nParámetros Funcionales (Familia Rk):")
            print(f"  Rpk: {Rpk:.3f} µm\t(Altura de picos que se desgastan)")
            print(f"  Rk: {Rk:.3f} µm\t(Profundidad del núcleo funcional)")
            print(f"  Rvk: {Rvk:.3f} µm\t(Profundidad de valles para lubricante)")
            print(f"  Mr1: {Mr1:.2f} %\t(Material ratio inferior)")
            print(f"  Mr2: {Mr2:.2f} %\t(Material ratio superior)")

            # --- Generación de Gráficas de Perfil ---
            plt.figure(figsize=(12, 6)); plt.plot(eje_x, perfil_primario, color='black', lw=1)
            plt.title('Perfil Primario'); plt.xlabel('Distancia (mm)'); plt.ylabel('Altura (µm)')
            plt.grid(True); plt.savefig('perfil_primario.png')
            print("Gráfica 'perfil_primario.png' guardada correctamente.")

            plt.figure(figsize=(12, 6)); plt.plot(eje_x, perfil_rugosidad, color='blue', lw=1)
            plt.title('Perfil de Rugosidad'); plt.xlabel('Distancia (mm)'); plt.ylabel('Altura (µm)')
            plt.grid(True); plt.savefig('perfil_rugosidad.png')
            print("Gráfica 'perfil_rugosidad.png' guardada correctamente.")
            
            print("\nAnálisis finalizado.")