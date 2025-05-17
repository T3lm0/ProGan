import argparse
import os
import torch
from torchvision.utils import save_image
from progan_model import Generator
from math import log2
import numpy as np
from PIL import Image

Z_DIM = 512

def generate_images(model, output_dir, num_images):
    """Generate images using the GAN model and save them to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    step = int(log2(256 / 4))
    for i in range(num_images):
        with torch.no_grad():
            noise = torch.randn((1, Z_DIM, 1, 1), device='cpu')
            generated_image = model(noise, 1, step)
            array = generated_image.detach().cpu().numpy()[0][0]
            scaled_array = ((array + 1) * 127.5).astype(np.uint8)
            gray_image = Image.fromarray(scaled_array, 'L')
            gray_image.save(os.path.join(output_dir, f"image_{i + 1}.png"))
            
def load_model(checkpoint_path, model, map_location=torch.device('cpu')):
    print(f"=> Loading checkpoint from '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint['state_dict'])
    
    print("✅ Model loaded successfully.")

def main():
    parser = argparse.ArgumentParser(description="Generate images using a GAN model.")
    parser.add_argument("model_path", type=str, help="Path to the GAN model file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the generated images.")
    parser.add_argument("num_images", type=int, help="Number of images to generate.")
    args = parser.parse_args()
    # Initialize the model
    model = Generator(z_dim=Z_DIM, in_channels=512, img_channels=1)  
    
    load_model(args.model_path, model)
    # Set the model to evaluation mode
    generate_images(model, args.output_dir, args.num_images)

if __name__ == "__main__":
    main()