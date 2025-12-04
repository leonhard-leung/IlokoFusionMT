"""
eval.py

This script evaluates a trained Iloko ↔ English Neural Machine Translation (NMT) model.

It loads the latest (best) checkpoint saved during training and computes several standard
MT evaluation metrics, including:
- BLEU (corpus-level)
- ChrF (Character-level F-score)
- Exact Match Accuracy

Usage:
    `python src/eval.py`
"""

import torch
from model import BaseNMT
from tqdm import tqdm
import config
from src.datamodule import TranslationDataModule
from utils import load_latest_checkpoint
from sacrebleu.metrics import BLEU, CHRF

def evaluate_metrics(model, dataloader, tokenizer, device):
    """
    Computes BLEU, ChrF, and Exact Match accuracy for the given model.

    :param model: The trained NMT model to evaluate
    :param dataloader: DataLoader providing source and target sequence.
    :param tokenizer: Tokenizer used to decode generated and reference outputs.
    :param device: Torch device (CPU or CUDA) for running model inference.
    :return: Tuple containing BLEU score, ChrF score, and Exact Match accuracy
    """
    model.eval()
    bleu = BLEU()
    chrf = CHRF()

    preds = []
    refs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Metric Evaluation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            source_sentences = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            pred_texts = model.generate_text(
                input_sentence=source_sentences,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=config.MAX_SEQ_LEN
            )

            label_texts = []
            for seq in labels:
                seq = [x for x in seq.tolist() if x != -100]
                label_texts.append(tokenizer.decode(seq, skip_special_tokens=True))

            preds.extend(pred_texts)
            refs.extend(label_texts)

    bleu_score = bleu.corpus_score(preds, [refs]).score
    chrf_score = chrf.corpus_score(preds, [refs]).score

    exact = sum([1 for p, r in zip(preds, refs) if p.strip() == r.strip()])
    exact_acc = exact / len(preds)

    return bleu_score, chrf_score, exact_acc

def main():
    """
    Loads the trained model checkpoint, prepares evaluation data, and computes
    the final evaluation metrics on the held-out evaluation split.

    This script is typically run after training to generate results for benchmark
    comparisons, research, and model reports.

    :return: None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load best model and tokenizer
    checkpoint_folder = config.CHECKPOINT_DIR / ("LexiconPointerNMT" if config.USE_POINTER else "BaseNMT")
    model, tokenizer, use_pointer, epoch = load_latest_checkpoint(
        checkpoint_folder,
        base_model_class=BaseNMT,
        device=device,
        dropout_rate=config.DROPOUT_RATE
    )

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

    # data loader for evaluation
    evaluation_loader = data_module.evaluation_dataloader()

    # ======== Evaluate Best Checkpoint ========
    print("\nLoading best checkpoint for final evaluation...")
    bleu_score, chrf_score, exact_acc = evaluate_metrics(
        model, evaluation_loader, tokenizer, device
    )

    print("\n===== FINAL METRICS (BEST MODEL) =====")
    print(f"BLEU Score : {bleu_score:.2f}")
    print(f"ChrF Score : {chrf_score:.2f}")
    print(f"Exact Match: {exact_acc * 100:.2f}%")
    print("=====================================")

if __name__ == "__main__":
    main()