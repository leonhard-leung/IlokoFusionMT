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

import os
import pandas as pd
import torch
from torch.optim import AdamW
from transformers import T5Tokenizer
from model import BaseNMT, LexiconPointerNMT
from tqdm import tqdm
import config
from src.datamodule import TranslationDataModule


def train_epoch(model, dataloader, optimizer, device):
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

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        train_bar.set_postfix({"loss": loss.item()})

    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, device):
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

            loss = outputs.loss
            total_loss += loss.item()

            val_bar.set_postfix({"loss": loss.item()})

        return total_loss / len(dataloader)


def save_checkpoint(model, tokenizer, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to: {save_dir}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = T5Tokenizer.from_pretrained(config.MODEL_NAME)

    if config.USE_POINTER:
        lexicon_df = pd.read_csv(config.LEXICON_CLEANED_CSV)
        lexicon = dict(zip(lexicon_df['Iloko'], lexicon_df["English"]))

        base = BaseNMT(model_name=config.MODEL_NAME)
        model = LexiconPointerNMT(base, lexicon, tokenizer)
        print("Using LexiconPointerNMT")
    else:
        model = BaseNMT(model_name=config.MODEL_NAME)
        print("Using BaseNMT")

    model.to(device)

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

    train_loader = data_module.train_dataloader()
    validation_loader = data_module.validation_dataloader()

    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    for epoch in range(config.NUM_EPOCHS):
        print(f"===================================")
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate_epoch(model, validation_loader, device)

        print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
        print(f"===================================")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, tokenizer, config.SAVE_DIR + f"-{epoch + 1}")

    print("Training Complete...")

if __name__ == "__main__":
    main()