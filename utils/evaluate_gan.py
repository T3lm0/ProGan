import torch
import numpy as np
from PIL import Image
import os
import random
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# --- Configuración ---
SOURCE_REAL_IMAGES_PATH = '/home/telmo/Escritorio/TFG/Codigo/datos/outDatNodulesCropped/0'  # Carpeta de imágenes reales
SOURCE_GENERATED_IMAGES_PATH = 'generated_images'                              # Carpeta de imágenes generadas
NUM_REAL_IMAGES_FOR_EVAL = 1000                                                # Cuántas imágenes reales usar
NUM_GENERATED_IMAGES_FOR_EVAL = 1000                                           # Cuántas imágenes generadas usar
BATCH_SIZE = 32                                                                # Tamaño del lote para procesamiento
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'                        # GPU si está disponible

# Transformación estándar para redes Inception (299x299 y normalización 0–1)
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor()
])

# --- Función para seleccionar rutas de imágenes aleatorias ---
def get_random_image_paths(source_directory, num_images_to_select):
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    all_image_paths = [os.path.join(source_directory, f)
                       for f in os.listdir(source_directory)
                       if f.lower().endswith(image_extensions)
                       and os.path.isfile(os.path.join(source_directory, f))]
    return random.sample(all_image_paths, min(len(all_image_paths), num_images_to_select))

# --- Generador por lotes de imágenes transformadas a tensores ---
def image_paths_to_batches(image_paths, batch_size):
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = []
        for path in batch_paths:
            try:
                with Image.open(path) as img:
                    img = img.convert('RGB')
                    img_tensor = transform(img)
                    images.append(img_tensor)
            except Exception as e:
                print(f"Error cargando imagen {path}: {e}")
        if images:
            yield torch.stack(images)

# --- Cálculo de FID ---
def calculate_fid_score(real_paths, gen_paths):
    """
    FID (Frechet Inception Distance):
    Compara distribuciones estadísticas entre imágenes reales y generadas,
    midiendo la distancia entre sus activaciones en una red InceptionV3.
    Valores más bajos indican mayor similitud visual y estadística.
    """
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)
    for real_batch in image_paths_to_batches(real_paths, BATCH_SIZE):
        fid.update((real_batch * 255).to(torch.uint8).to(DEVICE), real=True)
    for gen_batch in image_paths_to_batches(gen_paths, BATCH_SIZE):
        fid.update((gen_batch * 255).to(torch.uint8).to(DEVICE), real=False)
    return fid.compute().item()

# --- Cálculo de Inception Score ---
def calculate_inception_score(gen_paths):
    """
    IS (Inception Score):
    Evalúa la calidad y diversidad de las imágenes generadas.
    Imágenes de buena calidad tienden a tener distribuciones de clases bien definidas.
    Valores más altos indican mejor calidad y diversidad.
    """
    is_metric = InceptionScore(normalize=True).to(DEVICE)
    for gen_batch in image_paths_to_batches(gen_paths, BATCH_SIZE):
        is_metric.update((gen_batch * 255).to(torch.uint8).to(DEVICE))
    mean, std = is_metric.compute()
    return mean.item(), std.item()

# --- Cálculo de LPIPS ---
def calculate_lpips_score(real_paths, gen_paths):
    """
    LPIPS (Learned Perceptual Image Patch Similarity):
    Mide similitud perceptiva entre pares de imágenes (real vs generada).
    Basado en diferencias en activaciones de redes profundas entrenadas.
    Valores más bajos indican mayor similitud visual perceptual.
    """
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(DEVICE)
    min_len = min(len(real_paths), len(gen_paths))
    paired_paths = zip(real_paths[:min_len], gen_paths[:min_len])

    for i in range(0, min_len, BATCH_SIZE):
        real_batch, gen_batch = [], []
        for rp, gp in list(paired_paths)[i:i+BATCH_SIZE]:
            try:
                with Image.open(rp) as r_img, Image.open(gp) as g_img:
                    r_tensor = transform(r_img.convert('RGB'))
                    g_tensor = transform(g_img.convert('RGB'))
                    real_batch.append(r_tensor)
                    gen_batch.append(g_tensor)
            except Exception as e:
                print(f"Error cargando imágenes para LPIPS: {e}")
        if real_batch and gen_batch:
            real_tensor = torch.stack(real_batch).to(DEVICE)
            gen_tensor = torch.stack(gen_batch).to(DEVICE)
            lpips.update(real_tensor, gen_tensor)
    return lpips.compute().item()

# --- Ejecución Principal ---
if __name__ == "__main__":
    print(f"Usando dispositivo: {DEVICE}")

    # Selección aleatoria de imágenes
    print("Seleccionando imágenes reales y generadas...")
    real_paths = get_random_image_paths(SOURCE_REAL_IMAGES_PATH, NUM_REAL_IMAGES_FOR_EVAL)
    gen_paths = get_random_image_paths(SOURCE_GENERATED_IMAGES_PATH, NUM_GENERATED_IMAGES_FOR_EVAL)

    print(f"Imágenes reales seleccionadas: {len(real_paths)}")
    print(f"Imágenes generadas seleccionadas: {len(gen_paths)}")

    if len(real_paths) < 1000 or len(gen_paths) < 1000:
        print("⚠️  Se recomienda usar al menos 1000 imágenes para resultados fiables.")

    print("\n--- Cálculo de Métricas ---")

    # FID
    fid = calculate_fid_score(real_paths, gen_paths)
    print(f"\n✅ FID Score: {fid:.4f}")

    # IS
    mean_is, std_is = calculate_inception_score(gen_paths)
    print(f"✅ Inception Score: {mean_is:.4f} ± {std_is:.4f}")

    # LPIPS
    lpips = calculate_lpips_score(real_paths, gen_paths)
    print(f"✅ LPIPS Score: {lpips:.4f}")

    print("\n✅ Evaluación completada.")
