import cv2
import torch
from math import log2
import os
import torch
import random
import numpy as np
import os
import torchvision
import torch.nn as nn
from torchvision.utils import save_image
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
from progan_model import Generator, Discriminator
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from math import log2
from tqdm import tqdm
import json
import torch.multiprocessing as mp
from datetime import datetime

mp.set_start_method('spawn', force=True) # To avoid fork issues with DataLoader

torch.backends.cudnn.benchmarks = True

START_TRAIN_AT_IMG_SIZE = 512
CURRENT_IMG_SIZE = 512
DATASET = '/home/telmo/outDat/'
CHECKPOINT_GEN = '/home/telmo/train_models/gan2/generator_size_256_59.pth'
CHECKPOINT_CRITIC = '/home/telmo/train_models/gan2/critic_size_256_59.pth'

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = True
LEARNING_RATE = 1e-4
LEARNING_RATE_CRITIC = 3e-4
BATCH_SIZES = [64, 64, 32, 32, 16, 16, 32, 16]  # Reducido para tamaños grandes
CHANNELS_IMG = 1
Z_DIM = 512
IN_CHANNELS = 512
CRITIC_ITERATIONS = 9
LAMBDA_GP = 20
PROGRESSIVE_EPOCHS = [10, 10, 15, 20, 20, 40, 60, 80, 80]  # Más épocas para tamaños grandes
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 0
START = 0 # Epoch to start training


def plot_to_tensorboard(
    writer, loss_critic, loss_gen, real, fake, tensorboard_step
):
    writer.add_scalar("Loss Critic", loss_critic, global_step=tensorboard_step)

    with torch.no_grad():
        img_grid_real = torchvision.utils.make_grid(real[:8], normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake[:8], normalize=True)
        writer.add_image("Real", img_grid_real, global_step=tensorboard_step)
        writer.add_image("Fake", img_grid_fake, global_step=tensorboard_step)


def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    mixed_scores = critic(interpolated_images, alpha, train_step)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location="cuda:1", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_examples(gen, steps, truncation=0.7, n=100):
    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            noise = torch.tensor(truncnorm.rvs(-truncation, truncation, size=(1, Z_DIM, 1, 1)), device=DEVICE, dtype=torch.float32)
            img = gen(noise, alpha, steps)
            save_image(img*0.5+0.5, f"saved_examples/img_{i}.png")
    gen.train()
    


def get_loader(image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Asegurar tamaño correcto
        transforms.Grayscale(num_output_channels=1),  # Convertir a escala de grises
        transforms.ToTensor(), # Convertir a tensor
        transforms.Normalize([0.5], [0.5]),  # Normalización [-1,1]
    ])
    batch_size = BATCH_SIZES[int(log2(image_size / 4))]
    dataset = datasets.ImageFolder(root=DATASET, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return loader, dataset
""" Training of ProGAN using WGAN-GP loss"""


def train_fn(
    critic,
    gen,
    loader,
    dataset,
    step,
    alpha,
    opt_critic,
    opt_gen,
    tensorboard_step,
    writer,
    scaler_gen,
    scaler_critic,
):
    loop = tqdm(loader, leave=True)
    avg_loss_critic, avg_loss_gen = 0, 0
    total_batches = 0
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(DEVICE)
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)] <-> min -E[critic(real)] + E[critic(fake)]
        # which is equivalent to minimizing the negative of the expression
        noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(DEVICE)

        with torch.amp.autocast(device_type='cuda:1', dtype=torch.float32):
            fake = gen(noise, alpha, step)
            critic_real = critic(real, alpha, step)
            critic_fake = critic(fake.detach(), alpha, step)
            gp = gradient_penalty(critic, real, fake, alpha, step, device=DEVICE)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + LAMBDA_GP * gp
                + (0.001 * torch.mean(critic_real ** 2))
            )

        opt_critic.zero_grad()
        scaler_critic.scale(loss_critic).backward()
        scaler_critic.step(opt_critic)
        scaler_critic.update()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        with torch.amp.autocast(device_type='cuda:1', dtype=torch.float32):
            gen_fake = critic(fake, alpha, step)
            loss_gen = -torch.mean(gen_fake)

        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        # Update alpha and ensure less than 1
        alpha += cur_batch_size / (
            (PROGRESSIVE_EPOCHS[step] * 0.1) * len(dataset) # zmiana z 0.5 na 0.1
        )
        alpha = min(alpha, 1)

        if batch_idx % 500 == 0:
            with torch.no_grad():
                fixed_fakes = gen(FIXED_NOISE, alpha, step) * 0.5 + 0.5
            plot_to_tensorboard(
                writer,
                loss_critic.item(),
                loss_gen.item(),
                real.detach(),
                fixed_fakes.detach(),
                tensorboard_step,
            )
            tensorboard_step += 1

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
            loss_gen=loss_gen.item(),
        )
            
        total_batches += 1
        avg_loss_critic += loss_critic.item()
        avg_loss_gen += loss_gen.item()
    avg_loss_critic /= total_batches
    avg_loss_gen /= total_batches
    return tensorboard_step, alpha, avg_loss_critic, avg_loss_gen


if __name__ == "__main__":
    
    gen = Generator(
    Z_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG
    ).to(DEVICE)
    critic = Discriminator(
        Z_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG
    ).to(DEVICE)
    print(IN_CHANNELS, CHANNELS_IMG)
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(
        critic.parameters(), lr=LEARNING_RATE_CRITIC, betas=(0.0, 0.99)
    )
    scaler_critic = torch.amp.GradScaler()
    scaler_gen = torch.amp.GradScaler()

    writer = SummaryWriter(f"logs/gan2")
    if LOAD_MODEL:
        load_checkpoint(
            CHECKPOINT_GEN, gen, opt_gen, LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_CRITIC, critic, opt_critic, LEARNING_RATE_CRITIC,
        )
    gen.train()
    critic.train()

    tensorboard_step = 0
    step = int(log2(START_TRAIN_AT_IMG_SIZE / 4))
    for num_epochs in PROGRESSIVE_EPOCHS[step:]:
        # alpha = 1e-5
        alpha = 1
        img_size = 4 * 2 ** step
        CURRENT_IMG_SIZE = img_size
        loader, dataset = get_loader(img_size)
        print(f"\nEntrenando en tamaño {img_size}x{img_size}")
        print(f"Batch size: {BATCH_SIZES[step]}")
        print(f"Épocas programadas: {num_epochs}")
        log_file = os.path.join("logs/gan2", "train_log.txt")
        with open(log_file, "a") as f:
            f.write(f"Image size: {img_size}\n"
                    f"\tEpocas {num_epochs}\n")
        total_avg_loss_gen = 0
        total_avg_loss_critic = 0
        for epoch in range(START, num_epochs):
            print(f"\nÉpoca [{epoch+1}/{num_epochs}] - Tamaño {img_size}")
            
            tensorboard_step, alpha, avg_loss_critic, avg_loss_gen = train_fn(
                critic,
                gen,
                loader,
                dataset,
                step,
                alpha,
                opt_critic,
                opt_gen,
                tensorboard_step,
                writer,
                scaler_gen,
                scaler_critic,
            )

            
            for i in range(3):
                x = torch.randn((1, Z_DIM, 1, 1), device=DEVICE)
                z = gen(x, 1, steps=step)
                assert z.shape == (1, 1, img_size, img_size)
                out = critic(z, alpha=alpha, steps=step)
                print(out.shape)
                assert out.shape == (1, 1)
                print(f"Success! At img size: {img_size}")
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.imshow(z.detach().cpu().numpy()[0][0], cmap='gray')
                os.makedirs('train_imgs/gan2', exist_ok=True)
                plt.savefig(f'train_imgs/gan2/mass__size_{img_size}_epoch_{epoch}_{i}.png', bbox_inches='tight', pad_inches=0)

            if SAVE_MODEL and (epoch % 3 == 0 or epoch % 10 == 0 or epoch == num_epochs - 1):
                print("Saving model...")
                os.makedirs('train_models', exist_ok=True)
                generator_file = os.path.join('train_models/gan2', f"generator_size_{img_size}_{epoch}.pth")
                critic_file = os.path.join('train_models/gan2', f"critic_size_{img_size}_{epoch}.pth")
                save_checkpoint(gen, opt_gen, filename=generator_file)
                save_checkpoint(critic, opt_critic, filename=critic_file)
            # Accumulate average losses for critic and generator
            total_avg_loss_critic += avg_loss_critic
            total_avg_loss_gen += avg_loss_gen
            torch.cuda.empty_cache()
            # Calculate and print overall average losses after all epochs
            with open(log_file, "a") as f:
                f.write(
                    f"\t[Epoch {epoch+1}/{num_epochs}] "
                    f"Loss Critic: {avg_loss_critic:.4f} | "
                    f"Loss Gen: {avg_loss_gen:.4f} | "
                    f"LR Critic: {opt_critic.param_groups[0]['lr']:.2e} | "
                    f"LR Gen: {opt_gen.param_groups[0]['lr']:.2e} | "
                    f"GP λ: {LAMBDA_GP}\n"
                )        
        overall_avg_loss_critic = total_avg_loss_critic / num_epochs
        overall_avg_loss_gen = total_avg_loss_gen / num_epochs
        
        # Save training log
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"[{timestamp}] Summary for image size {img_size}x{img_size} ({num_epochs} epochs)\n")
            f.write(f"→ Avg Loss Critic: {overall_avg_loss_critic:.4f}\n")
            f.write(f"→ Avg Loss Gen:    {overall_avg_loss_gen:.4f}\n")
            f.write(f"→ LR Critic:       {opt_critic.param_groups[0]['lr']:.2e}\n")
            f.write(f"→ LR Gen:          {opt_gen.param_groups[0]['lr']:.2e}\n")
            f.write(f"→ GP λ:            {LAMBDA_GP}\n")
            f.write("="*80 + "\n\n")            
        step += 1