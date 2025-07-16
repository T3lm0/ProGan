import os
import argparse
from tqdm import tqdm
from PIL import Image, ImageEnhance

def augment_image(image_path, output_dir):
    img = Image.open(image_path).convert('RGB')
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    ext = os.path.splitext(image_path)[1]

    # Save original
    img.save(os.path.join(output_dir, f"{base_name}_orig{ext}"))

    # 1. Horizontal Flip
    img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    img_flip.save(os.path.join(output_dir, f"{base_name}_flip{ext}"))

    # 2. Slight Brightness Increase
    enhancer = ImageEnhance.Brightness(img)
    img_bright = enhancer.enhance(1.2)
    img_bright.save(os.path.join(output_dir, f"{base_name}_bright{ext}"))

    # 3. Slight Contrast Increase
    enhancer = ImageEnhance.Contrast(img)
    img_contrast = enhancer.enhance(1.2)
    img_contrast.save(os.path.join(output_dir, f"{base_name}_contrast{ext}"))

def main():
    parser = argparse.ArgumentParser(description="Augment images in a folder with simple transformations.")
    parser.add_argument("folder", type=str, help="Path to the folder containing images.")
    args = parser.parse_args()

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    files = [f for f in os.listdir(args.folder) if f.lower().endswith(image_extensions)]

    for fname in tqdm(files, desc="Augmenting images"):
        fpath = os.path.join(args.folder, fname)
        augment_image(fpath, args.folder)

if __name__ == "__main__":
    main()