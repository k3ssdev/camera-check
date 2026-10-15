#!/usr/bin/env python3
"""
Sensor Inspector para pruebas de cámara (formato JPEG, TIFF o RAW .ORF)
"""

import argparse
from pathlib import Path
import cv2
import numpy as np


def load_image(file_path: Path) -> np.ndarray:
    """Carga imagen tanto en formatos estándar como RAW (.ORF)."""
    if file_path.suffix.lower() in [".orf", ".raw", ".dng"]:
        import rawpy
        with rawpy.imread(str(file_path)) as raw:
            # Post-procesado plano sin balance agresivo ni reducción de ruido
            rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    img = cv2.imread(str(file_path))
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {file_path}")
    return img


def check_hot_pixels(dark_img: np.ndarray, threshold: int = 40):
    """
    Analiza un dark frame buscando hot pixels o píxeles atascados.
    """
    gray = cv2.cvtColor(dark_img, cv2.COLOR_BGR2GRAY)
    
    # Calcular estadísticas de ruido de fondo
    mean_noise = np.mean(gray)
    std_noise = np.std(gray)
    
    # Un hot pixel se define por superar un umbral de corte absoluto o relativo
    hot_mask = gray > max(threshold, mean_noise + 5 * std_noise)
    y_coords, x_coords = np.where(hot_mask)
    num_hot = len(x_coords)

    print("\n--- [ANÁLISIS DE DARK FRAME: HOT PIXELS] ---")
    print(f"Ruido base medio: {mean_noise:.2f} (std: {std_noise:.2f})")
    print(f"Píxeles sospechosos / calientes detectados: {num_hot}")

    if num_hot > 0:
        print("Coordenadas destacadas (X, Y) y valores:")
        for i in range(min(num_hot, 10)):
            x, y = x_coords[i], y_coords[i]
            print(f"  -> ({x}, {y}) | Valor: {gray[y, x]}")
        if num_hot > 10:
            print(f"  ... y {num_hot - 10} más.")
    else:
        print("✓ Sensor impecable en térmicos: no hay hot pixels destacados.")


def check_dead_pixels_and_dust(flat_img: np.ndarray):
    """
    Analiza un flat frame buscando píxeles muertos y sombras de motas de polvo.
    """
    gray = cv2.cvtColor(flat_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Píxeles muertos (valor 0 o cercano a 0 en una toma iluminada)
    dead_mask = gray <= 3
    y_dead, x_dead = np.where(dead_mask)
    num_dead = len(x_dead)

    print("\n--- [ANÁLISIS DE FLAT FIELD: PÍXELES MUERTOS] ---")
    print(f"Píxeles muertos detectados: {num_dead}")
    if num_dead > 0:
        for i in range(min(num_dead, 10)):
            print(f"  -> ({x_dead[i]}, {y_dead[i]})")
    else:
        print("✓ Sin píxeles muertos detectados.")

    # 2. Detección de polvo (sombras circulares)
    # Suavizado amplio para estimar la iluminación base (viñeteo/gradiente)
    blurred = cv2.GaussianBlur(gray, (101, 101), 0)
    
    # División para aplanar el campo y quedarnos solo con variaciones locales
    normalized = cv2.divide(gray, blurred, scale=255)
    
    # Las motas aparecen como caídas locales de luz (oscurecimiento relativo)
    dust_mask = normalized < 240  # Caída de más del ~6% respecto al fondo local
    
    # Filtrado morfológico para eliminar ruido aislado de 1 px y buscar manchas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_dust = cv2.morphologyEx(dust_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_dust)
    
    # Excluir el fondo (label 0)
    dust_spots = [
        (centroids[i], stats[i, cv2.CC_STAT_AREA])
        for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] >= 4  # Al menos 4 píxeles agrupados
    ]

    print("\n--- [ANÁLISIS DE FLAT FIELD: POLVO Y MANCHAS] ---")
    print(f"Motas de polvo relevantes detectadas: {len(dust_spots)}")
    if dust_spots:
        print("Ubicaciones principales (X, Y) y tamaño aprox en píxeles:")
        for (cx, cy), area in dust_spots[:10]:
            print(f"  -> ({int(cx)}, {int(cy)}) | Área: {area} px²")
        if len(dust_spots) > 10:
            print(f"  ... y {len(dust_spots) - 10} manchas menores.")
    else:
        print("✓ Sensor limpio: no se aprecian sombras de suciedad significativas.")


def main():
    parser = argparse.ArgumentParser(description="Auditor de sensor OM System / Micro 4/3")
    parser.add_argument("--dark", type=Path, help="Ruta a la foto con tapa (dark frame)")
    parser.add_argument("--flat", type=Path, help="Ruta a la foto f/22 blanca (flat field)")
    args = parser.parse_args()

    if not args.dark and not args.flat:
        parser.error("Debes proporcionar al menos una imagen con --dark o --flat.")

    if args.dark:
        print(f"Cargando Dark Frame: {args.dark.name}...")
        img_dark = load_image(args.dark)
        check_hot_pixels(img_dark)

    if args.flat:
        print(f"\nCargando Flat Field: {args.flat.name}...")
        img_flat = load_image(args.flat)
        check_dead_pixels_and_dust(img_flat)


if __name__ == "__main__":
    main()
