"""
datamodule.py

This module defines the `TranslationDataset` as well as the `TranslationDataModule` for handling
parallel corpus data for neural machine translation (NMT).

Outputs:
1. Training DataLoader
2. Validation DataLoader
3. Testing DataLoader

Notes:
    - Supports bidirectional translation (source ↔ target).
    - Handles train, test, and validation splits.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from preprocessing import load_csv
from sklearn.model_selection import train_test_split

class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer_src, tokenizer_tgt, max_len=128):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.max_len = max_len

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src = self.tokenizer_src(
            self.src_texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len
        )
        tgt = self.tokenizer_tgt(
            self.tgt_texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len
        )

        input_ids = torch.tensor(src["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(src["attention_mask"], dtype=torch.long)
        labels = torch.tensor(tgt["input_ids"], dtype=torch.long)

        labels[labels == self.tokenizer_tgt.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class TranslationDataModule:
    def __init__(
            self,
            csv_path,
            tokenizer_src,
            tokenizer_tgt,
            batch_size=32,
            test_split=0.2,
            val_split=0.1,
            max_len=128,
            num_workers=4,
            pin_memory=True
    ):
        self.csv_path = csv_path
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.batch_size = batch_size
        self.test_split = test_split
        self.val_split = val_split
        self.max_len = max_len
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self):
        # load dataset
        df = load_csv(self.csv_path)

        iloko_texts = df["Iloko"].tolist()
        english_texts = df["English"].tolist()

        # duplicate dataset for bidirectional translation
        src_texts = iloko_texts + english_texts
        tgt_texts = english_texts + iloko_texts

        # split for train_test and validation
        src_train_test, src_val, tgt_train_test, tgt_val = train_test_split(
            src_texts, tgt_texts, test_size=self.val_split, random_state=42
        )

        src_train, src_test, tgt_train, tgt_test = train_test_split(
            src_train_test, tgt_train_test, test_size=self.test_split, random_state=42
        )

        self.train_dataset = TranslationDataset(
            src_train, tgt_train, self.tokenizer_src, self.tokenizer_tgt, max_len=self.max_len
        )

        self.test_dataset = TranslationDataset(
            src_test, tgt_test, self.tokenizer_src, self.tokenizer_tgt, max_len=self.max_len
        )

        self.val_dataset = TranslationDataset(
            src_val, tgt_val, self.tokenizer_src, self.tokenizer_tgt, max_len=self.max_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )