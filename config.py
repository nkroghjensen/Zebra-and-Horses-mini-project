import torch

"""
This configuration script defines all parameters and settings used by train.py 
Controling dataset loading, model selection, optimization behaviour, and overall training execution.

Contains:
- Data paths
- Model selection
- Hyperparameters
- Device configuration
- Logging / W&B settings

This file is used by both models.py and train.py.
"""

#______________________________________________________________________________________________________________________________
#
# DATA DIRECTORIES
#______________________________________________________________________________________________________________________________

# This file defines all configuration settings and hyperparameters used across the project.
# Modifying these values allows you to control training behaviour, model performance, and reproducibility.
DATA_DIR = "datasets"
TRAIN_DIR = "datasets/train/"
VALID_DIR = "datasets/val"
TEST_DIR = "datasets/test"  # final evaluation dataset

# Number of classes in the dataset (binary classification: horse vs. zebra)
NUM_CLASSES = 2


#______________________________________________________________________________________________________________________________
#
# MODEL CONFIGURATION
#______________________________________________________________________________________________________________________________

# Select which model implementation to use:
# "resnet_pretrained"  torchvision ResNet-18 with ImageNet weights (transfer learning)
# "resnet_custom"      manually implemented ResNet-18 architecture
MODEL_TYPE = "resnet_pretrained"


#______________________________________________________________________________________________________________________________
#
# TRAINING HYPERPARAMETERS
#______________________________________________________________________________________________________________________________

# Batch size determines how many images are processed in each training step.
# Larger batches yield more stable gradient estimates and more reliable accuracy metrics,
# but require more GPU memory. 128 is a practical balance.
BATCH_SIZE = 128

# Number of epochs defines how long the model is allowed to train.
# More epochs can improve performance until the model saturates.
NUM_EPOCHS = 30

# The learning rate controls the magnitude of updates to the network parameters.
# A small LR learns slowly but stably; a large LR risks overshooting and unstable training.
LEARNING_RATE = 1e-3

# Weight decay is a regularization technique that penalizes large weights
# to reduce overfitting and encourage the model to learn balanced features.
WEIGHT_DECAY = 1e-4

# Optimizer type: AdamW decouples weight decay from gradient updates,
# resulting in more reliable regularization behaviour.
OPTIMIZER = "adamw"

# Learning rate scheduler reduces the LR when validation performance stagnates.
# Patience defines how long to wait; factor defines the multiplicative reduction.
LR_SCHEDULER = "plateau"
LR_PATIENCE = 3
LR_FACTOR = 0.25

# Number of worker processes used by DataLoader for faster data loading.
NUM_WORKERS = 4


#______________________________________________________________________________________________________________________________
#
# IMAGE & DATA AUGMENTATION SETTINGS
#______________________________________________________________________________________________________________________________

# Target input resolution for the model.
# Lower sizes train faster; higher sizes retain more spatial detail.
IMAGE_SIZE = 224

# ImageNet normalization ensures images match the expected distribution
# of pretrained models and stabilizes training.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Data augmentation settings.
# These transforms improve generalization by applying randomised perturbations during training.
RANDOM_HORIZONTAL_FLIP = 0.5
RANDOM_ROTATION = 15

# Colour jitter kept conservative to avoid destroying class-critical colour information.
COLOR_JITTER_BRIGHTNESS = 0.1
COLOR_JITTER_CONTRAST = 0.15
COLOR_JITTER_SATURATION = 0.05


#______________________________________________________________________________________________________________________________
#
# SAVING / OUTPUT
#______________________________________________________________________________________________________________________________

# Directory where models and results will be saved.
SAVE_DIR = "resultat"

# Base filename for saving model checkpoints.
MODEL_SAVE_NAME = "model_resnet18.pth"


#______________________________________________________________________________________________________________________________
#
# DEVICE & REPRODUCIBILITY
#______________________________________________________________________________________________________________________________

# Device selection. Automatically uses GPU if available; otherwise CPU.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Random seed ensures fully reproducible training when combined with fixed data and hyperparameters.
RANDOM_SEED = 42

# Controls how often training progress is printed (in batches).
PRINT_FREQUENCY = 10


#______________________________________________________________________________________________________________________________
#
# Weights & Biases (experiment tracking)
#______________________________________________________________________________________________________________________________

# Enable or disable W&B experiment logging.
USE_WANDB = True

# IMPORTANT: Never store real API keys in the config file.
# Leave this as None. W&B automatically reads your key from the environment:
#     export WANDB_API_KEY="your_secret_key"
WANDB_API_KEY = None

# W&B project metadata.
WANDB_PROJECT = "horses-vs-zebras"
WANDB_ENTITY = None  # optional: your username

#______________________________________________________________________________________________________________________________
#
# EVALUATION IN TRAIN
#______________________________________________________________________________________________________________________________

EVALUATE_DURING_TRAINING = True
SAVE_CONFUSION_MATRIX = True

