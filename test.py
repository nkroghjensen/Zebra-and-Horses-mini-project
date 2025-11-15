import os
import random
from typing import Tuple, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

import config
import models

# Optional: wandb (Weights & Biases) for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


"""
test.py

Evaluate a trained model on the held-out test set.

Implements:
- Reproducible evaluation (same seed setup as train.py)
- Rebuilds the architecture via models.build_model_from_config()
- Loads the best checkpoint saved by train.py
- Runs inference on the test split (datasets/test)
- Computes:
    * Overall test accuracy
    * Confusion matrix
    * Precision / recall / F1 per class (classification report)
- Optionally logs results + confusion matrix to Weights & Biases

The evaluation pattern (looping over a data iterator, accumulating
correct predictions, etc.) mirrors the style used in "Dive into Deep
Learning" helper utilities such as `evaluate_accuracy_gpu`.
"""

#_______________________________________________________________________________________
#
# REPRODUCIBILITY
#_______________________________________________________________________________________

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Matches the setup from train.py so that training + test runs
    are comparable and deterministic given fixed data.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#_______________________________________________________________________________________
#
# DATA: TEST TRANSFORMS + DATALOADER
#_______________________________________________________________________________________

def build_eval_transform() -> transforms.Compose:
    """
    Build evaluation transform: resize + tensor + normalization.

    No data augmentation here – we want a clean, unbiased estimate
    of generalisation on the test set.
    """
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])


def build_test_loader() -> tuple[DataLoader, List[str]]:
    """
    Create DataLoader for the test split and return class names.

    Uses torchvision.datasets.ImageFolder, assuming directory structure:
        datasets/
            train/...
            val/...
            test/
                hest/
                zebra/
    """
    if not os.path.isdir(config.TEST_DIR):
        raise FileNotFoundError(
            f"Test directory not found: {config.TEST_DIR}\n"
            "Make sure your dataset is split into train/val/test."
        )

    eval_transform = build_eval_transform()

    test_dataset = datasets.ImageFolder(
        root=config.TEST_DIR,
        transform=eval_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    class_names = test_dataset.classes  # e.g. ['hest', 'zebra']

    return test_loader, class_names


#_______________________________________________________________________________________
#
# MODEL LOADING
#_______________________________________________________________________________________

def load_trained_model() -> nn.Module:
    """
    Rebuild the model architecture and load trained weights.

    The checkpoint path mirrors train.py:
        SAVE_DIR / f"{MODEL_TYPE}_{MODEL_SAVE_NAME}"
    For example:
        resultat/resnet_pretrained_model_resnet18.pth
    """
    # Build the model architecture from config (pretrained/custom)
    model = models.build_model_from_config()
    model.to(config.DEVICE)

    # Path to the best model saved during training
    checkpoint_path = os.path.join(
        config.SAVE_DIR,
        f"{config.MODEL_TYPE}_{config.MODEL_SAVE_NAME}",
    )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train the model first with train.py so the weights are saved."
        )

    state_dict = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded trained model from: {checkpoint_path}")
    print(f"Evaluating on device: {config.DEVICE}")

    return model


#_______________________________________________________________________________________
#
# EVALUATION LOOP
#_______________________________________________________________________________________

def run_inference(
    model: nn.Module,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run inference on the whole test set.

    Returns:
        all_targets: numpy array of ground-truth labels
        all_preds:   numpy array of predicted labels
        accuracy:    scalar accuracy over the whole test set
    """
    all_targets: list[int] = []
    all_preds: list[int] = []

    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

            correct += (preds == targets).sum().item()
            total += targets.size(0)

    accuracy = correct / total if total > 0 else 0.0

    return (
        np.array(all_targets, dtype=np.int64),
        np.array(all_preds, dtype=np.int64),
        accuracy,
    )


#_______________________________________________________________________________________
#
# CONFUSION MATRIX VISUALISATION
#_______________________________________________________________________________________

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    save_path: str | None = None,
) -> None:
    """
    Plot (optionally normalised) confusion matrix with class names.

    This gives a "global explainability" view: where does the model
    confuse heste og zebraer? Very similar to how many D2L examples
    inspect misclassifications when analysing classifiers.
    """
    if normalize:
        cm = cm.astype(np.float32)
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm = np.divide(
            cm,
            cm_sum,
            out=np.zeros_like(cm),
            where=cm_sum != 0,
        )

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation="nearest")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix (test set)")

    # Write values into the cells
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            ax.text(
                j,
                i,
                format(value, fmt),
                ha="center",
                va="center",
                color="white" if value > thresh else "black",
                fontsize=9,
            )

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Confusion matrix figure saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()


#_______________________________________________________________________________________
#
# W&B INITIALISATION (OPTIONAL)
#_______________________________________________________________________________________

def init_wandb_for_test() -> bool:
    """
    Optionally start a Weights & Biases run for evaluation.

    Mirrors the integration described i din handlingsplan:
    log test accuracy + confusion matrix for nem rapportering.
    """
    if not getattr(config, "USE_WANDB", False):
        return False

    if not WANDB_AVAILABLE:
        print("wandb is not installed. Skipping W&B logging.")
        return False

    if config.WANDB_API_KEY is not None:
        os.environ["WANDB_API_KEY"] = str(config.WANDB_API_KEY)

    wandb.init(
        project=config.WANDB_PROJECT,
        entity=config.WANDB_ENTITY,
        name=f"evaluation_{config.MODEL_TYPE}",
        config={
            "stage": "test",
            "model_type": config.MODEL_TYPE,
            "batch_size": config.BATCH_SIZE,
        },
    )

    return True


#_______________________________________________________________________________________
#
# MAIN
#_______________________________________________________________________________________

def main() -> None:
    print("===== Test evaluation =====")
    print(f"Using device: {config.DEVICE}")

    # 1) Reproducibility
    set_seed(config.RANDOM_SEED)

    # 2) Data (test loader)
    test_loader, class_names = build_test_loader()
    num_test_samples = len(test_loader.dataset)
    print(f"Test samples: {num_test_samples}")
    print(f"Detected classes: {class_names}")

    # 3) Model
    model = load_trained_model()

    # 4) Optional W&B run
    use_wandb = init_wandb_for_test()

    # 5) Inference
    y_true, y_pred, test_acc = run_inference(model, test_loader)
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

    # 6) Detailed metrics
    print("\nClassification report (per class):")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
        )
    )

    # 7) Confusion matrix (raw + figure)
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix (counts):")
    print(cm)

    # Save figure (normalised version) to SAVE_DIR
    cm_fig_path = os.path.join(
        config.SAVE_DIR,
        f"{config.MODEL_TYPE}_confusion_matrix.png",
    )
    plot_confusion_matrix(
        cm,
        class_names=class_names,
        normalize=True,
        save_path=cm_fig_path,
    )

    # 8) Optional W&B logging
    if use_wandb:
        # Log scalar metrics
        wandb.log(
            {
                "test_accuracy": test_acc,
            }
        )

        # Log confusion matrix as interactive W&B plot
        try:
            cm_plot = wandb.plot.confusion_matrix(
                preds=y_pred.tolist(),
                y_true=y_true.tolist(),
                class_names=class_names,
            )
            wandb.log({"confusion_matrix": cm_plot})
        except Exception as e:
            print(f"Could not log W&B confusion matrix: {e}")

        # Also log saved image (nice for reports)
        try:
            wandb.log({"confusion_matrix_image": wandb.Image(cm_fig_path)})
        except Exception as e:
            print(f"Could not log confusion matrix image: {e}")

        wandb.finish()

    print("\nEvaluation finished. Ready to include results in the report.")


if __name__ == "__main__":
    main()
