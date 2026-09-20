import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# ============================================================
# Configuration
# ============================================================

SEED = 42
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

LABEL_TO_ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID_TO_LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Load data
# ============================================================

def load_data(file_path):
    df = pd.read_parquet(file_path)

    # Keep only the three main sentiment classes
    df = df[
        df["label"].isin(
            ["negative", "neutral", "positive"]
        )
    ].copy()

    df["label_id"] = df["label"].map(LABEL_TO_ID)

    return df


# ============================================================
# Group split
# ============================================================

def split_by_sentence(df):
    """
    Split by full sentence instead of individual aspect rows.

    This prevents the same sentence from appearing in both
    training and test data.
    """

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=SEED,
    )

    train_index, test_index = next(
        splitter.split(
            df,
            groups=df["text"],
        )
    )

    train_df = df.iloc[train_index].reset_index(drop=True)
    test_df = df.iloc[test_index].reset_index(drop=True)

    return train_df, test_df


# ============================================================
# Dataset
# ============================================================

class ABSADataset(torch.utils.data.Dataset):

    def __init__(self, dataframe, tokenizer):
        self.dataframe = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        sentence = str(row["text"])
        aspect = str(row["span"])

        # Aspect-conditioned input
        model_input = (
            f"Sentence: {sentence} "
            f"Aspect: {aspect}"
        )

        encoding = self.tokenizer(
            model_input,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        return {
            "input_ids":
                encoding["input_ids"].squeeze(0),

            "attention_mask":
                encoding["attention_mask"].squeeze(0),

            "labels":
                torch.tensor(
                    row["label_id"],
                    dtype=torch.long,
                ),
        }


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, data_loader, device):

    model.eval()

    all_predictions = []
    all_labels = []

    total_loss = 0.0

    with torch.no_grad():

        for batch in data_loader:

            input_ids = batch["input_ids"].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()

            predictions = torch.argmax(
                outputs.logits,
                dim=-1,
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    average_loss = (
        total_loss / len(data_loader)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
    )

    return (
        average_loss,
        accuracy,
        macro_f1,
        all_labels,
        all_predictions,
    )


# ============================================================
# Main
# ============================================================

def main():

    set_seed(SEED)

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    data_path = (
        project_root
        / "data"
        / "laptop"
        / "train-00000-of-00001.parquet"
    )

    output_dir = (
        project_root
        / "models"
        / "laptop_group_split"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("Aspect-Based Sentiment Analysis")
    print("Laptop In-Domain Experiment")
    print("=" * 60)

    print(f"\nDevice: {device}")

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    print("\nLoading data...")

    df = load_data(data_path)

    print(
        f"Total labeled samples: {len(df)}"
    )

    print("\nFull label distribution:")
    print(df["label"].value_counts())

    train_df, test_df = split_by_sentence(df)

    print("\n===== Dataset Split =====")

    print(
        f"Training samples: {len(train_df)}"
    )

    print(
        f"Test samples: {len(test_df)}"
    )

    print(
        f"Training sentences: "
        f"{train_df['text'].nunique()}"
    )

    print(
        f"Test sentences: "
        f"{test_df['text'].nunique()}"
    )

    # Safety check for leakage
    train_sentences = set(train_df["text"])
    test_sentences = set(test_df["text"])

    overlap = (
        train_sentences
        .intersection(test_sentences)
    )

    print(
        f"Sentence overlap: {len(overlap)}"
    )

    if len(overlap) != 0:
        raise RuntimeError(
            "Data leakage detected."
        )

    print("\nTraining label distribution:")
    print(train_df["label"].value_counts())

    print("\nTest label distribution:")
    print(test_df["label"].value_counts())

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # Dataset / DataLoader
    # --------------------------------------------------------

    train_dataset = ABSADataset(
        train_df,
        tokenizer,
    )

    test_dataset = ABSADataset(
        test_df,
        tokenizer,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nLoading DistilBERT...")

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=3,
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
        )
    )

    model.to(device)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n===== Training =====")

    best_macro_f1 = -1.0

    for epoch in range(EPOCHS):

        model.train()

        total_train_loss = 0.0

        for batch_number, batch in enumerate(
            train_loader,
            start=1,
        ):

            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels = batch[
                "labels"
            ].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss

            loss.backward()

            optimizer.step()

            total_train_loss += loss.item()

            if batch_number % 20 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} "
                    f"| Batch "
                    f"{batch_number}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f}"
                )

        average_train_loss = (
            total_train_loss
            / len(train_loader)
        )

        (
            test_loss,
            accuracy,
            macro_f1,
            _,
            _,
        ) = evaluate(
            model,
            test_loader,
            device,
        )

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"Train Loss: "
            f"{average_train_loss:.4f}"
        )

        print(
            f"Test Loss: "
            f"{test_loss:.4f}"
        )

        print(
            f"Accuracy: "
            f"{accuracy:.2%}"
        )

        print(
            f"Macro F1: "
            f"{macro_f1:.4f}"
        )

        # Save best model
        if macro_f1 > best_macro_f1:

            best_macro_f1 = macro_f1

            model.save_pretrained(
                output_dir
            )

            tokenizer.save_pretrained(
                output_dir
            )

            print(
                "Best model updated."
            )

        print("-" * 60)

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    print(
        "\nLoading best model for "
        "final evaluation..."
    )

    best_model = (
        AutoModelForSequenceClassification
        .from_pretrained(output_dir)
    )

    best_model.to(device)

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    (
        final_loss,
        final_accuracy,
        final_macro_f1,
        true_labels,
        predictions,
    ) = evaluate(
        best_model,
        test_loader,
        device,
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(
        f"Test samples: "
        f"{len(test_df)}"
    )

    print(
        f"Accuracy: "
        f"{final_accuracy:.2%}"
    )

    print(
        f"Macro F1: "
        f"{final_macro_f1:.4f}"
    )

    print(
        f"Test Loss: "
        f"{final_loss:.4f}"
    )

    print("\nClassification Report:")

    report = classification_report(
        true_labels,
        predictions,
        labels=[0, 1, 2],
        target_names=[
            "negative",
            "neutral",
            "positive",
        ],
        digits=4,
        zero_division=0,
    )

    print(report)

    print(
        f"Best model saved to:\n"
        f"{output_dir}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()