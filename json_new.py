import json
import shutil
import os
import sys
from tqdm import tqdm  # Importar tqdm

if len(sys.argv) != 5:
    print(len(sys.argv))
    print('Usage: python3 json_new.py <json_path> <source_dir_equalized> <destination_dir_for_equalized> <new_json_path>')
    exit()

json_path = sys.argv[1]
src_equ_dir = sys.argv[2]
dst_equ_dir = sys.argv[3]
new_json_path = sys.argv[4]

def create_json():
    # Cargar el JSON original
    with open(json_path, "r") as file:
        data = json.load(file)

    # Obtener lista de imágenes ecualizadas 
    print("Escaneando imágenes ecualizadas...")
    equ_images = [f for f in tqdm(os.listdir(src_equ_dir)) if f.endswith("_equ_.png")]

    # Crear una copia del JSON original
    combined_data = data.copy()

    print("\nProcesando imágenes...")
    for equ_image in tqdm(equ_images, desc="Copiando imágenes"):
        base_name = equ_image.replace("_equ_.png", ".dcm")
        
        if base_name in data:
            new_key = base_name.replace(".dcm", "_equ_.dcm")
            new_image_path = data[base_name]["image"].replace(".png", "_equ_.png")

            combined_data[new_key] = {
                "image": new_image_path,
                "label": data[base_name]["label"]
            }

            # Copiar la imagen ecualizada al destino
            src_image_path = os.path.join(src_equ_dir, equ_image)
            dst_image_path = os.path.join(dst_equ_dir, equ_image)

            if not os.path.exists(dst_equ_dir):
                os.makedirs(dst_equ_dir)

            shutil.copy(src_image_path, dst_image_path)
        else:
            tqdm.write(f"⚠️ Imagen ecualizada {equ_image} no tiene entrada en el JSON original.")

    # Guardar el JSON combinado
    with open(new_json_path, "w") as file:
        json.dump(combined_data, file, indent=4)

    print(f"\n✅ JSON combinado guardado en: {new_json_path}")
    print(f"Total imágenes procesadas: {len(equ_images)}")

if __name__ == '__main__':
    create_json()