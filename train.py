"""
In this script the training of a given model is handled. 

Implements:
- Reproducible training loop for a given model
- Support for pretrained and custom ResNet-18 via config.MODEL_TYPE
- Data loading with ImageFolder + augmentation
- Training + validation per epoch
- Optional Weights & Biases logging
- Saving best model weights
"""

import os
import random
from typing import Tuple, Dict

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config
import models

# Optional: wandb (Weights & Biases) for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ============================================================
# REPRODUCIBILITY: set random seeds
# ============================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # More deterministic behavior (can slow things down slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET & DATALOADERS
# ============================================================

def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Build training and validation/test transforms.

    Theory links:
    - Resize → common input size  (IMAGE_SIZE)
    - Data augmentation → improves generalization
    - Normalization → matches distribution expected by ResNet
    """
    train_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=config.RANDOM_HORIZONTAL_FLIP),
        transforms.RandomRotation(degrees=config.RANDOM_ROTATION),
        transforms.ColorJitter(
            brightness=config.COLOR_JITTER_BRIGHTNESS,
            contrast=config.COLOR_JITTER_CONTRAST,
            saturation=config.COLOR_JITTER_SATURATION,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])

    return train_transform, eval_transform


def build_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader | None]:
    """
    Create DataLoaders for train, validation and test splits.

    Uses:
    - torchvision.datasets.ImageFolder (directory = class)
    - DataLoader with shuffle for train, no shuffle for eval
    """
    train_t, eval_t = build_transforms()

    train_dataset = datasets.ImageFolder(root=config.TRAIN_DIR, transform=train_t)
    val_dataset = datasets.ImageFolder(root=config.VALID_DIR, transform=eval_t)

    # Test is optional (may not exist in some setups)
    test_dataset = None
    if os.path.isdir(config.TEST_DIR):
        test_dataset = datasets.ImageFolder(root=config.TEST_DIR, transform=eval_t)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val   samples: {len(val_dataset)}")
    if test_loader is not None:
        print(f"Test  samples: {len(test_dataset)}")

    print(f"Detected classes: {train_dataset.classes}")

    return train_loader, val_loader, test_loader


# ============================================================
# MODEL, LOSS, OPTIMIZER, SCHEDULER
# ============================================================

def build_model_loss_optimizer() -> Tuple[nn.Module, nn.Module, optim.Optimizer, optim.lr_scheduler._LRScheduler | None]:
    """
    Initialize:
    - Model (from models.build_model_from_config)
    - CrossEntropy loss (for classification)
    - Optimizer (Adam / AdamW)
    - Optional LR scheduler
    """
    # Build model (pretrained or custom) – already moved to DEVICE
    model = models.build_model_from_config()

    # Standard multi-class classification loss
    criterion = nn.CrossEntropyLoss()

    # Select optimizer
    if config.OPTIMIZER.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

    # Learning rate scheduler (optional)
    scheduler = None
    if config.LR_SCHEDULER == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.LR_FACTOR,
            patience=config.LR_PATIENCE,
        )

    return model, criterion, optimizer, scheduler


# ============================================================
# TRAIN / EVAL HELPERS
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
) -> Tuple[float, float]:
    """
    Train for a single epoch.

    Steps:
    - model.train()
    - forward pass
    - compute loss
    - backward pass (backpropagation)
    - optimizer step (update weights)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(config.DEVICE)
        targets = targets.to(config.DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)               # forward
        loss = criterion(outputs, targets)    # compute loss

        loss.backward()                       # backpropagation
        optimizer.step()                      # update parameters

        running_loss += loss.item() * inputs.size(0)

        # Compute accuracy
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        if (batch_idx + 1) % config.PRINT_FREQUENCY == 0:
            print(
                f"  [Batch {batch_idx+1}/{len(loader)}] "
                f"Train Loss: {loss.item():.4f}"
            )

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """
    Evaluation loop (no gradient computation).

    Used for:
    - Validation per epoch
    - Final test evaluation
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# ============================================================
# OPTIONAL: TEST EVALUATION + CONFUSION MATRIX
# ============================================================

def evaluate_on_test(model: nn.Module, test_loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
    """
    Run a final evaluation on the test set.

    If SAVE_CONFUSION_MATRIX is True, compute confusion matrix.
    """
    from sklearn.metrics import confusion_matrix

    model.eval()
    all_targets = []
    all_preds = []

    test_loss, test_acc = evaluate(model, test_loader, criterion)

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    results = {
        "test_loss": test_loss,
        "test_acc": test_acc,
    }

    print(f"\nFinal TEST results: Loss={test_loss:.4f}, Acc={test_acc*100:.2f}%")

    if config.SAVE_CONFUSION_MATRIX:
        cm = confusion_matrix(all_targets, all_preds)
        print("Confusion matrix:")
        print(cm)
        results["confusion_matrix"] = cm

    return results


# ============================================================
# W&B INITIALIZATION
# ============================================================

def init_wandb() -> bool:
    """Initialize Weights & Biases if enabled and available."""
    if not config.USE_WANDB:
        return False
    if not WANDB_AVAILABLE:
        print("wandb is not installed. Skipping W&B logging.")
        return False

    # Login if API key is provided (env-var recommended)
    if config.WANDB_API_KEY is not None:
        wandb.login(key=config.WANDB_API_KEY)
    else:
        # wandb will look for WANDB_API_KEY in environment
        try:
            wandb.login()
        except Exception as e:
            print(f"wandb login failed: {e}")
            return False

    wandb.init(
        project=config.WANDB_PROJECT,
        entity=config.WANDB_ENTITY,
        config={
            "model_type": config.MODEL_TYPE,
            "batch_size": config.BATCH_SIZE,
            "epochs": config.NUM_EPOCHS,
            "learning_rate": config.LEARNING_RATE,
            "weight_decay": config.WEIGHT_DECAY,
            "optimizer": config.OPTIMIZER,
            "scheduler": config.LR_SCHEDULER,
            "image_size": config.IMAGE_SIZE,
        },
    )

    return True


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def main() -> None:
    print(f"Using device: {config.DEVICE}")
    set_seed(config.RANDOM_SEED)

    # Data
    train_loader, val_loader, test_loader = build_dataloaders()

    # Model, loss, optimizer, scheduler
    model, criterion, optimizer, scheduler = build_model_loss_optimizer()

    # W&B
    use_wandb = init_wandb()
    if use_wandb:
        wandb.watch(model, log="gradients", log_freq=10)

    best_val_acc = 0.0
    best_model_path = os.path.join(
        config.SAVE_DIR,
        f"{config.MODEL_TYPE}_{config.MODEL_SAVE_NAME}",
    )
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    # Training epochs
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}% "
            f"| Val Loss={val_loss:.4f}, Val Acc={val_acc*100:.2f}%"
        )

        # Scheduler step (ReduceLROnPlateau uses validation loss)
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # W&B logging
        if use_wandb:
            current_lr = optimizer.param_groups[0]["lr"]
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": current_lr,
                }
            )

        # Track best model (based on validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  → New best model saved to: {best_model_path}")

    print(f"\nTraining finished. Best Val Acc: {best_val_acc*100:.2f}%")

    # Load best model before final evaluation (optional)
    if os.path.isfile(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
        print(f"Loaded best model from: {best_model_path}")

    # Optional: final evaluation on test set
    if config.EVALUATE_DURING_TRAINING and test_loader is not None:
        evaluate_on_test(model, test_loader, criterion)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
