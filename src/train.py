"""
train.py

This script trains a bidirectional Iloko ↔ English Neural Machine Translation (NMT) model.

This script supports training two types of models:
1. BaseNMT: A standard sequence-to-sequence model using T5.
2. LexiconPointerNMT: Extends the BaseNMT with a lexicon pointer mechanism for handling
out-of-vocabulary (OOV) words during translation.

Features:
- Bidirectional translation
- Configurable via the `config.py` file
- Automatic train/validation split
- Checkpoint saving for best validation loss
- GPU support if available

Requirements:
- Parallel CSV file with columns: "Iloko", "English"
- (Optional) Lexicon CSV for LexiconPointerNMT with columns: "Iloko", "English"

Usage:
    python src/train.py
"""

import pandas as pd
import torch
from torch.optim import AdamW
from transformers import T5Tokenizer
from model import BaseNMT, LexiconPointerNMT
from tqdm import tqdm
import config
from src.datamodule import TranslationDataModule
from utils import load_latest_checkpoint, save_checkpoint

def train_epoch(model, dataloader, optimizer, device):
    """
    Performs one epoch of training on the given model using the provided dataloader.

    :param model: PyTorch model to train
    :param dataloader: DataLoader for training data
    :param optimizer: Optimizer for updating model weights
    :param device: Torch device (CPU or CUDA)
    :return: Average training loss for the epoch
    """

    model.train()
    total_loss = 0

    train_bar = tqdm(dataloader, desc="Training")
    for batch in train_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if config.USE_POINTER:
            outputs, pointer_prob = outputs

        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        train_bar.set_postfix({"loss": loss.item()})

    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, device):
    """
    Evaluates the model on the validation dataset and computes the average loss.

    :param model: PyTorch model to evaluate
    :param dataloader: DataLoader for validation data
    :param device: Torch device (CPU or CUDA)
    :return: Average validation loss for the epoch
    """

    model.eval()
    total_loss = 0

    with torch.no_grad():

        val_bar = tqdm(dataloader, desc="Validation")
        for batch in val_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            if config.USE_POINTER:
                outputs, pointer_prob = outputs

            loss = outputs.loss
            total_loss += loss.item()

            val_bar.set_postfix({"loss": loss.item()})

        return total_loss / len(dataloader)

def main():
    """
    Main training routine for the NMT model.

    Handles model instantiation, checkpoint loading, data preparation,
    training, validation, and saving checkpoints.

    :return: None
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = T5Tokenizer.from_pretrained(config.MODEL_NAME)

    # model instantiation
    base_model = BaseNMT(
        model_name=config.MODEL_NAME,
        dropout=config.DROPOUT,
        attention_dropout=config.ATTENTION_DROPOUT,
        activation_dropout=config.ACTIVATION_DROPOUT,
    )

    if config.USE_POINTER:
        lexicon_df = pd.read_csv(config.LEXICON_CLEANED_CSV)
        lexicon = dict(zip(lexicon_df['Iloko'], lexicon_df["English"]))

        model = LexiconPointerNMT(base_model, lexicon, tokenizer)
        print("Using LexiconPointerNMT")
    else:
        model = base_model
        print("Using BaseNMT")

    model.to(device)

    # load from a checkpoint
    checkpoint_folder = config.CHECKPOINT_DIR / ("LexiconPointerNMT" if config.USE_POINTER else "BaseNMT")

    try:
        model, start_epoch = load_latest_checkpoint(model, checkpoint_folder)
        print(f"Resuming training from epoch {start_epoch + 1}...")
    except FileNotFoundError:
        start_epoch = 0
        print("No checkpoint found, starting training from scratch...")

    # prepare data
    data_module = TranslationDataModule(
        csv_path=config.PARALLEL_CLEANED_CSV,
        tokenizer=tokenizer,
        batch_size=config.BATCH_SIZE,
        evaluation_split=config.EVALUATION_SPLIT,
        validation_split=config.VALIDATION_SPLIT,
        max_len=config.MAX_SEQ_LEN,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    data_module.setup()

    # data loader for training and validation
    train_loader = data_module.train_dataloader()
    validation_loader = data_module.validation_dataloader()

    # optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # training cycle
    best_val_loss = float("inf")
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print(f"===================================")
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate_epoch(model, validation_loader, device)

        print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
        print(f"===================================")

        # ======== checkpoint saving ========
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, tokenizer,
                checkpoint_folder,
                f"epoch-{epoch + 1}.pt"
            )

    print("Training Complete...")

if __name__ == "__main__":
    main()