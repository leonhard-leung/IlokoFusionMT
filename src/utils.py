import os
import torch
from pathlib import Path

def save_checkpoint(model, tokenizer, save_path, filename):
    os.makedirs(save_path, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "tokenizer": tokenizer.name_or_path,
    }

    if hasattr(model, "lexicon"):
        checkpoint["lexicon"] = model.lexicon

    torch.save(checkpoint, f"{save_path}/{filename}")
    print(f"Model saved to: {save_path}\\{filename}")

# def load_latest_checkpoint(model, tokenizer, base_dir):
