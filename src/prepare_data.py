"""
prepare_data.py

This script performs the preprocessing pipeline on the raw parallel corpus and lexicon CSV files.
It should be run **once** before training the NMT + Pointer Fusion model to generate the cleaned
CSV files located in `data/processed`.

Usage:
    - `python src/prepare_data.py` : terminal

Outputs:
    - `data/processed/parallel_cleaned.csv` : cleaned parallel corpus
    - `data/processed/lexicon_cleaned.csv` : cleaned lexicon

Notes:
    - This script uses functions from preprocessing.py to clean texts from the dataframes.
    - Do not run during training; train.py uses the processed CSV files which this script outputs.
"""

from preprocessing import load_csv, clean_dataframe, save_csv
import config

def main():
    parallel_df = load_csv(config.RAW_PARALLEL_CSV)
    lexicon_df = load_csv(config.RAW_LEXICON_CSV)

    parallel_df_cleaned = clean_dataframe(parallel_df, src_col="Iloko", tgt_col="English")
    lexicon_df_cleaned = clean_dataframe(lexicon_df, src_col="Iloko", tgt_col="English")

    save_csv(parallel_df_cleaned, config.PARALLEL_CLEANED_CSV)
    save_csv(lexicon_df_cleaned, config.LEXICON_CLEANED_CSV)

    print("Data cleaned and saved...")

if __name__ == "__main__":
    main()