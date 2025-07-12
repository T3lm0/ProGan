import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

def smart_crop_mammogram(image, tile_size=32, pixel_threshold=10, content_ratio=0.01, margin=550):
    """
    Recorta automáticamente la mama en la imagen, detectando de qué lado comienza (izq/der)
    y ajustando el recorte sin necesidad de reorientar (sin flip),
    manteniendo la relación de aspecto original de la imagen añadiendo relleno (padding).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h_original, w_original = gray.shape
    original_aspect_ratio = w_original / h_original

    # Initialize bounds to full image in case content is not found
    left_bound = 0
    right_bound = w_original

    # Evaluar contenido en extremos
    left_tile = gray[:, :tile_size]
    right_tile = gray[:, -tile_size:]
    left_ratio = np.count_nonzero(left_tile > pixel_threshold) / left_tile.size
    right_ratio = np.count_nonzero(right_tile > pixel_threshold) / right_tile.size

    num_tiles = w_original // tile_size

    # Caso 1: mama empieza en la izquierda (más contenido en izquierda o similar)
    if left_ratio >= right_ratio:
        # Buscar primer tile con contenido desde la izquierda
        found_left_bound = False
        for i in range(num_tiles):
            tile = gray[:, i * tile_size : (i + 1) * tile_size]
            non_black_ratio = np.count_nonzero(tile > pixel_threshold) / tile.size
            if non_black_ratio > content_ratio:
                left_bound = max(0, i * tile_size - margin)
                found_left_bound = True
                break
        if not found_left_bound: # If no content found, revert to default
            left_bound = 0

        # Buscar último tile con contenido desde la derecha
        found_right_bound = False
        for i in range(num_tiles - 1, -1, -1):
            tile = gray[:, i * tile_size : (i + 1) * tile_size]
            non_black_ratio = np.count_nonzero(tile > pixel_threshold) / tile.size
            if non_black_ratio > content_ratio:
                right_bound = min(w_original, (i + 1) * tile_size + margin)
                found_right_bound = True
                break
        if not found_right_bound: # If no content found, revert to default
            right_bound = w_original

    # Caso 2: mama empieza en la derecha (más contenido en derecha)
    else:
        # Buscar primer tile con contenido desde la derecha
        found_right_bound = False
        for i in range(num_tiles - 1, -1, -1):
            tile = gray[:, i * tile_size : (i + 1) * tile_size]
            non_black_ratio = np.count_nonzero(tile > pixel_threshold) / tile.size
            if non_black_ratio > content_ratio:
                right_bound = min(w_original, (i + 1) * tile_size + margin)
                found_right_bound = True
                break
        if not found_right_bound: # If no content found, revert to default
            right_bound = w_original

        # Buscar último tile con contenido desde la izquierda
        found_left_bound = False
        for i in range(num_tiles):
            tile = gray[:, i * tile_size : (i + 1) * tile_size]
            non_black_ratio = np.count_nonzero(tile > pixel_threshold) / tile.size
            if non_black_ratio > content_ratio:
                left_bound = max(0, i * tile_size - margin)
                found_left_bound = True
                break
        if not found_left_bound: # If no content found, revert to default
            left_bound = 0


    # Recortar horizontalmente usando los límites detectados
    cropped_horizontal = image[:, left_bound:right_bound]

    # --- Mantener la relación de aspecto original ---
    current_h, current_w = cropped_horizontal.shape[0], cropped_horizontal.shape[1]
    
    # Calcular la altura deseada para mantener la relación de aspecto original
    target_h = int(current_w / original_aspect_ratio)
    
    # Si la altura actual es menor que la altura deseada, añadir padding vertical
    if target_h > current_h:
        pad_h = target_h - current_h
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        
        # Determine padding color based on image channels
        if len(image.shape) == 3: # Color image
            border_color = (0, 0, 0) # Black for RGB
        else: # Grayscale image
            border_color = 0 # Black for grayscale

        cropped_final = cv2.copyMakeBorder(cropped_horizontal, pad_top, pad_bottom, 0, 0,
                                         cv2.BORDER_CONSTANT, value=border_color)
    else:
        # If current height is already greater or equal, no vertical padding needed.
        # However, if target_h is significantly smaller, we might consider cropping vertically,
        # but for maintaining original aspect ratio after horizontal crop, padding is generally preferred.
        # For simplicity, we just use the horizontally cropped image if no padding is needed.
        cropped_final = cropped_horizontal

    return cropped_final

def crop_images_in_folder(input_folder, output_folder, **crop_kwargs):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

    for filename in tqdm(image_files, desc="Cropping images"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        image = cv2.imread(input_path)
        if image is not None:
            # Check if image is grayscale and read it as such if needed
            if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            
            cropped = smart_crop_mammogram(image, **crop_kwargs)
            cv2.imwrite(output_path, cropped)
        else:
            print(f"Warning: Could not read {input_path}")
                
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Crop mammogram images in a folder.")
    parser.add_argument("input_folder", help="Path to the input folder with images")
    parser.add_argument("output_folder", help="Path to the output folder for cropped images")
    parser.add_argument("--tile_size", type=int, default=32, help="Tile size for scanning (default: 32)")
    parser.add_argument("--pixel_threshold", type=int, default=10, help="Pixel threshold for content (default: 10)")
    parser.add_argument("--content_ratio", type=float, default=0.01, help="Content ratio threshold (default: 0.01)")
    parser.add_argument("--margin", type=int, default=550, help="Margin to add to crop bounds (default: 550)")

    args = parser.parse_args()

    crop_images_in_folder(
        args.input_folder,
        args.output_folder,
        tile_size=args.tile_size,
        pixel_threshold=args.pixel_threshold,
        content_ratio=args.content_ratio,
        margin=args.margin
    )