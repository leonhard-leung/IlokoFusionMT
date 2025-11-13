"""
utils.py

This module contains utility functions used to load and save model checkpoints
"""

import os
import torch
from pathlib import Path


def save_checkpoint(model, tokenizer, save_path, filename):
    """
    Saves the model state, tokenizer reference, and optional lexicon to a checkpoint file.

    :param model: PyTorch model whose state_dict will be saved
    :param tokenizer: Tokenizer used with the model
    :param save_path: Directory path where the checkpoint will be saved
    :param filename: Name of the checkpoint file
    :return: None
    """
    os.makedirs(save_path, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "tokenizer": tokenizer.name_or_path,
    }

    if hasattr(model, "lexicon"):
        checkpoint["lexicon"] = model.lexicon

    torch.save(checkpoint, f"{save_path}/{filename}")
    print(f"Model saved to: {save_path}\\{filename}")


def load_latest_checkpoint(model, checkpoint_dir):
    """
    Loads the most recent checkpoint from the specified directory and updates the model weights.
    Also restores the lexicon if it exists in the checkpoint and the model supports it.

    :param model: PyTorch model to load the checkpoint into
    :param checkpoint_dir: Directory containing checkpoint files
    :return: Tuple of (model with loaded weights, last epoch number)
    :raises: FileNotFoundError: if the directory or checkpoint files do not exist
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"{checkpoint_dir} does not exist")

    pt_files = sorted(checkpoint_dir.glob("*.pt"), key=lambda x: int(x.stem.split("-")[-1]))
    if not pt_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    latest_file = pt_files[-1]
    checkpoint = torch.load(latest_file, map_location=torch.device("cpu"))

    model.load_state_dict(checkpoint["model_state_dict"])

    if "lexicon" in checkpoint and hasattr(model, "lexicon"):
        model.lexicon = checkpoint["lexicon"]

    print(f"Loaded checkpoint from: {latest_file}")

    epoch = int(latest_file.stem.split("-")[-1])

    return model, epoch