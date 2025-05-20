import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
import torch
from torchvision.utils import save_image
from progan_model import Generator # Assuming this is in the same directory
from math import log2
import numpy as np
from PIL import Image

# Set CustomTkinter theme and color
ctk.set_appearance_mode("System")  # Can be "System" (default), "Dark", "Light"
ctk.set_default_color_theme("blue") # Can be "blue", "green", "dark-blue"

def generate_images_gui(model, output_dir, num_images_str, progress_label, z_dim, resolution_step_str):
    """
    Generate images using the GAN model and save them to the output directory.
    This version includes GUI updates for progress and configurable Z_DIM and resolution step.
    """
    try:
        num_images = int(num_images_str)
        if num_images <= 0:
            raise ValueError("Number of images must be a positive integer.")
    except ValueError as e:
        messagebox.showerror("Invalid Input", f"Number of images must be an integer: {e}")
        return

    try:
        # Assuming the resolution step is an integer from the log2 of the target resolution
        # For a 256x256 image, the step is log2(256/4) = log2(64) = 6
        resolution_step = int(resolution_step_str)
        if resolution_step < 0: # Minimum step would typically be 0 for 4x4 or 1 for 8x8 etc.
             raise ValueError("Resolution step cannot be negative.")
    except ValueError as e:
        messagebox.showerror("Invalid Input", f"Invalid resolution step: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    # It's good practice to ensure permissions, though on Windows it might not be strictly necessary
    # and on Linux/macOS, it might be restrictive depending on umask.
    # Consider adjusting permissions if you encounter issues.
    try:
        os.chmod(output_dir, 0o777)
    except OSError as e:
        print(f"Warning: Could not set permissions for {output_dir}: {e}")

    # Ensure model is in evaluation mode
    model.eval()
    
    for i in range(num_images):
        with torch.no_grad():
            # Use the provided Z_DIM
            noise = torch.randn((1, z_dim, 1, 1), device='cpu')
            
            # The '1' in model(noise, 1, resolution_step) is the 'alpha' parameter.
            # The 'resolution_step' determines the current resolution being generated.
            # Make sure your Generator model expects these parameters.
            generated_image = model(noise, 1, resolution_step)
            
            # Convert tensor to PIL Image for saving
            array = generated_image.detach().cpu().numpy()[0][0] # Assuming grayscale output [C, H, W] -> [H, W] for grayscale
            scaled_array = ((array + 1) * 127.5).astype(np.uint8)
            gray_image = Image.fromarray(scaled_array, 'L') # 'L' mode for grayscale
            
            save_path = os.path.join(output_dir, f"image_{i + 1}.png")
            gray_image.save(save_path)
        
        progress_label.configure(text=f"Generating image {i + 1}/{num_images}...", text_color="orange")
        ctk.CTk.update_idletasks(progress_label) # Update the GUI
    
    progress_label.configure(text="Generation complete!", text_color="green")
    messagebox.showinfo("Success", f"Successfully generated {num_images} images in {output_dir}")

def load_model_gui(checkpoint_path, model, progress_label, map_location=torch.device('cpu')):
    """Load model with GUI updates."""
    progress_label.configure(text=f"Loading model from '{checkpoint_path}'...", text_color="blue")
    ctk.CTk.update_idletasks(progress_label)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model.load_state_dict(checkpoint['state_dict'])
        progress_label.configure(text="Model loaded successfully.", text_color="green")
        return True
    except Exception as e:
        messagebox.showerror("Model Loading Error", f"Failed to load model: {e}")
        progress_label.configure(text="Model loading failed.", text_color="red")
        return False

def main_gui():
    app = ctk.CTk()
    app.title("GAN Image Generator")
    app.geometry("550x650") # Adjust window size

    # --- Model Path ---
    model_frame = ctk.CTkFrame(app)
    model_frame.pack(pady=10, padx=20, fill="x")

    ctk.CTkLabel(model_frame, text="Model Path:").pack(side="left", padx=(10,0), pady=10)
    model_path_var = ctk.StringVar()
    model_entry = ctk.CTkEntry(model_frame, textvariable=model_path_var, width=300)
    model_entry.pack(side="left", padx=5, pady=10, expand=True, fill="x")

    def select_model_path():
        path = filedialog.askopenfilename(
            title="Select GAN Model File",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
        )
        if path:
            model_path_var.set(path)
    
    model_button = ctk.CTkButton(model_frame, text="Browse", command=select_model_path, width=80)
    model_button.pack(side="right", padx=5, pady=10)

    # --- Output Directory ---
    output_frame = ctk.CTkFrame(app)
    output_frame.pack(pady=10, padx=20, fill="x")

    ctk.CTkLabel(output_frame, text="Output Directory:").pack(side="left", padx=(10,0), pady=10)
    output_dir_var = ctk.StringVar()
    output_entry = ctk.CTkEntry(output_frame, textvariable=output_dir_var, width=300)
    output_entry.pack(side="left", padx=5, pady=10, expand=True, fill="x")

    def select_output_dir():
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            output_dir_var.set(path)

    output_button = ctk.CTkButton(output_frame, text="Browse", command=select_output_dir, width=80)
    output_button.pack(side="right", padx=5, pady=10)

    # --- Parameters Frame ---
    params_frame = ctk.CTkFrame(app)
    params_frame.pack(pady=10, padx=20, fill="x")

    # Z_DIM
    ctk.CTkLabel(params_frame, text="Z_DIM (Latent Vector Dim.):").grid(row=0, column=0, padx=10, pady=5, sticky="w")
    z_dim_var = ctk.StringVar(value="512") # Default Z_DIM
    z_dim_entry = ctk.CTkEntry(params_frame, textvariable=z_dim_var, width=100)
    z_dim_entry.grid(row=0, column=1, padx=10, pady=5, sticky="w")

    # Number of Images
    ctk.CTkLabel(params_frame, text="Number of Images:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
    num_images_var = ctk.StringVar(value="10") # Default value
    num_images_entry = ctk.CTkEntry(params_frame, textvariable=num_images_var, width=100)
    num_images_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")

    # Resolution Step
    # Explain what this means to the user or provide a more intuitive way to select resolution
    # For ProGAN, 'step' typically corresponds to log2(resolution / 4)
    # e.g., 4x4 -> step 0, 8x8 -> step 1, ..., 256x256 -> step 6, 1024x1024 -> step 8
    ctk.CTkLabel(params_frame, text="Resolution Step (e.g., 6 for 256x256):").grid(row=2, column=0, padx=10, pady=5, sticky="w")
    resolution_step_var = ctk.StringVar(value=str(int(log2(256/4)))) # Default to 256x256 resolution
    resolution_step_entry = ctk.CTkEntry(params_frame, textvariable=resolution_step_var, width=100)
    resolution_step_entry.grid(row=2, column=1, padx=10, pady=5, sticky="w")
    
    # --- Progress Label ---
    progress_label = ctk.CTkLabel(app, text="Ready", text_color="blue", font=("Roboto", 14))
    progress_label.pack(pady=20)

    # --- Generate Button ---
    def start_generation():
        model_path = model_path_var.get()
        output_dir = output_dir_var.get()
        num_images_str = num_images_var.get()
        resolution_step_str = resolution_step_var.get()

        try:
            current_z_dim = int(z_dim_var.get())
            if current_z_dim <= 0:
                raise ValueError("Z_DIM must be a positive integer.")
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Z_DIM must be an integer: {e}")
            return

        if not model_path:
            messagebox.showwarning("Missing Input", "Please select a model file.")
            return
        if not output_dir:
            messagebox.showwarning("Missing Input", "Please select an output directory.")
            return
        
        # Initialize the model with the current Z_DIM value
        # Ensure your Generator class can handle a variable z_dim correctly
        try:
            model = Generator(z_dim=current_z_dim, in_channels=512, img_channels=1)
            model.eval() # Set model to evaluation mode
        except Exception as e:
            messagebox.showerror("Model Initialization Error", f"Could not initialize Generator with Z_DIM={current_z_dim}: {e}")
            return

        if load_model_gui(model_path, model, progress_label):
            generate_images_gui(model, output_dir, num_images_str, progress_label, current_z_dim, resolution_step_str)

    generate_button = ctk.CTkButton(app, text="Generate Images", command=start_generation, 
                                    font=("Roboto", 16, "bold"), height=50)
    generate_button.pack(pady=20, padx=20, fill="x")

    app.mainloop()

if __name__ == "__main__":
    main_gui()