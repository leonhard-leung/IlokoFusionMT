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
            evaluation_split=0.1,
            validation_split=0.2,
            max_len=128,
            num_workers=4,
            pin_memory=True
    ):
        self.csv_path = csv_path
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.batch_size = batch_size
        self.evaluation_split = evaluation_split
        self.validation_split = validation_split
        self.max_len = max_len
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_dataset = None
        self.validation_dataset = None
        self.evaluation_dataset = None

    def setup(self):
        # load dataset
        df = load_csv(self.csv_path)

        iloko_texts = df["Iloko"].tolist()
        english_texts = df["English"].tolist()

        # duplicate dataset for bidirectional translation
        src_texts = iloko_texts + english_texts
        tgt_texts = english_texts + iloko_texts

        # split for train_val and evaluation
        src_train_val, src_eval, tgt_train_val, tgt_eval = train_test_split(
            src_texts, tgt_texts, test_size=self.evaluation_split
        )

        src_train, src_val, tgt_train, tgt_val = train_test_split(
            src_train_val, tgt_train_val, test_size=self.validation_split
        )

        self.train_dataset = TranslationDataset(
            src_train, tgt_train, self.tokenizer_src, self.tokenizer_tgt, max_len=self.max_len
        )

        self.validation_dataset = TranslationDataset(
            src_val, tgt_val, self.tokenizer_src, self.tokenizer_tgt, max_len=self.max_len
        )

        self.evaluation_dataset = TranslationDataset(
            src_eval, tgt_eval, self.tokenizer_src, self.tokenizer_tgt, max_len=self.max_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def validation_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def evaluation_dataloader(self):
        return DataLoader(
            self.evaluation_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )