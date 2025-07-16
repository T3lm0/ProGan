import torch

# Image size and dataset configuration
START_TRAIN_AT_IMG_SIZE = 4
CURRENT_IMG_SIZE = 4
DATASET = '/home/telmo/Escritorio/TFG/Codigo/datos/outDatNodulosCropped'

# Checkpoint and directory paths
CHECKPOINT_GEN = 'train_models/gan5Calc/generator_size_512_299.pth'
CHECKPOINT_CRITIC = 'train_models/gan5Calc/critic_size_512_299.pth'
LOG_DIR = "logs/gan5Calc/"
IMGS_DIR = "train_imgs/gan5Calc/"
MODELS_DIR = "train_models/gan5Calc/"

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model saving/loading
SAVE_MODEL = True
LOAD_MODEL = False

# Learning rates
LEARNING_RATE = 1e-3
LEARNING_RATE_CRITIC = 1e-3
LEARNING_RATES = [1e-3, 1e-3, 1e-3, 1e-4, 1e-3, 1e-4, 1e-4, 1e-4]
LEARNING_RATES_CRITIC = [1e-3, 1e-3, 1e-3, 1e-3, 1e-4, 3e-4, 3e-4, 3e-4]

# Batch sizes and channels
BATCH_SIZES = [128, 64, 32, 128, 64, 32, 28, 24]  # Reducido para tamaños grandes
CHANNELS_IMG = 1
Z_DIM = 512
IN_CHANNELS = 512

# Training hyperparameters
CRITIC_ITERATIONS = 20
LAMBDA_GP = 20
PROGRESSIVE_EPOCHS = [10, 10, 20, 25, 60, 100, 150, 400, 120]  # Más épocas para tamaños grandes

# Fixed noise for evaluation
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 8
START = 0 # Epoch to start training

