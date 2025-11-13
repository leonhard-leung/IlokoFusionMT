"""
preprocessing.py

This module contains reusable functions for text cleaning, dataframe preprocessing, and CSV
file handling. It is intended to be **imported** by scripts such as `prepare_data.py` for
preprocessing datasets.

Sections:
1. Text Cleaning: functions that removes whitespace, special characters, and lowercase text.
2. DataFrame Pipeline: function that applies the text cleaning functions to all entries in a dataframe.
3. File Handling: functions used to load and save CSV files.

Usage:
    import preprocessing

    df = preprocessing.load_csv(filepath)
    df = preprocessing.clean_dataframe(df)
"""

import re
import pandas as pd

# ======== text cleaning ========

def _whitespace_cleaning(text: str) -> str:
    if not isinstance(text, str):
        return text
    return re.sub(r"\s+", " ", text).strip()

def _remove_special_chars(text: str) -> str:
    if not isinstance(text, str):
        return text
    return re.sub(r"[^A-Za-z0-9.,!?'\-\s]+", " ", text)

def _to_lowercase(text: str) -> str:
    if not isinstance(text, str):
        return text
    return text.lower()

def clean_text(text: str) -> str:
    text = _to_lowercase(text)
    text = _remove_special_chars(text)
    text = _whitespace_cleaning(text)
    return text

# ======== dataframe pipeline ========

def clean_dataframe(
        df: pd.DataFrame,
        src_col: str = "Iloko",
        tgt_col: str = "English",
        drop_empty = True
) -> pd.DataFrame:
    df = df.copy()

    df[src_col] = df[src_col].astype(str).apply(clean_text)
    df[tgt_col] = df[tgt_col].astype(str).apply(clean_text)

    if drop_empty:
        df = df[(df[src_col].str.strip() != "") & (df[tgt_col].str.strip() != "")]

    df = df.reset_index(drop=True)
    return df


# ======== file handling ========

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)