"""
utils.py

This module contains utility functions used to load and save model checkpoints
"""

import os
import torch
from pathlib import Path
from model import LexiconPointerNMT
from transformers import T5Tokenizer

def save_checkpoint(model, tokenizer, save_path, filename, use_pointer=False):
    """
    Saves the model state, tokenizer name, lexicon, and pointer flag to a checkpoint file.

    :param model: PyTorch model whose state_dict will be saved
    :param tokenizer: Tokenizer used with the model
    :param save_path: Directory path where the checkpoint will be saved
    :param filename: Name of the checkpoint file
    :param use_pointer: Boolean indicating whether the model uses LexiconPointerNMT
    :return: None
    """
    os.makedirs(save_path, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "tokenizer_name": tokenizer.name_or_path,
        "use_pointer": use_pointer,
    }

    if hasattr(model, "lexicon"):
        checkpoint["lexicon"] = model.lexicon

    torch.save(checkpoint, f"{save_path}/{filename}")
    print(f"Model saved to: {save_path}\\{filename}")

def load_latest_checkpoint(checkpoint_dir, base_model_class=None, device="cpu", **model_kwargs):
    """
    Loads the most recent checkpoint from the specified directory and updates the model weights.
    Also restores the lexicon if it exists in the checkpoint and the model supports it.

    :param checkpoint_dir: Directory containing checkpoint files
    :param base_model_class: The class for BaseNMT (required if use_pointer=True)
    :param device: Device to map the model weights to (cpu or cuda)
    :param model_kwargs: Extra arguments to pass to base model constructor
    :return: Tuple of (model, tokenizer, use_pointer_flag, last_epoch)
    :raises: FileNotFoundError: if the directory or checkpoint files do not exist
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"{checkpoint_dir} does not exist")

    pt_files = sorted(checkpoint_dir.glob("*.pt"), key=lambda x: int(x.stem.split("-")[-1]))
    if not pt_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    latest_file = pt_files[-1]
    checkpoint = torch.load(latest_file, map_location=device)

    use_pointer = checkpoint.get("use_pointer", False)
    tokenizer = T5Tokenizer.from_pretrained(checkpoint["tokenizer_name"])

    if base_model_class is None:
        raise ValueError("base_model_class must be provided to load the model")

    base_model = base_model_class(**model_kwargs)

    if use_pointer:
        lexicon = checkpoint.get("lexicon", {})
        model = LexiconPointerNMT(base_model, lexicon, tokenizer)
    else:
        model = base_model

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    epoch = int(latest_file.stem.split("-")[-1])
    print(f"Loaded checkpoint from: {latest_file}")

    return model, tokenizer, use_pointer, epoch