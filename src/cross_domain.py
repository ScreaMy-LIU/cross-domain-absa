import pandas as pd
import numpy as np
import torch
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# ============================================================
# Configuration
# ============================================================

SEED = 42
MAX_LENGTH = 128
BATCH_SIZE = 32

LABEL_TO_ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}


# ============================================================
# Load Restaurant data
# ============================================================

def load_data(file_path):

    df = pd.read_parquet(file_path)

    # Use the same 3 classes as the Laptop experiment
    df = df[
        df["label"].isin(
            ["negative", "neutral", "positive"]
        )
    ].copy()

    df["label_id"] = df["label"].map(LABEL_TO_ID)

    return df


# ============================================================
# Create held-out Restaurant test set
# ============================================================

def create_restaurant_test_split(df):
    """
    Create the same sentence-grouped 80/20 split used for
    the in-domain experiments.

    Only the held-out 20% is used for cross-domain evaluation.
    """

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=SEED,
    )

    _, test_index = next(
        splitter.split(
            df,
            groups=df["text"],
        )
    )

    test_df = (
        df.iloc[test_index]
        .reset_index(drop=True)
    )

    return test_df


# ============================================================
# Dataset
# ============================================================

class ABSADataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
    ):
        self.dataframe = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        sentence = str(row["text"])
        aspect = str(row["span"])

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

def evaluate(
    model,
    data_loader,
    device,
):

    model.eval()

    all_predictions = []
    all_labels = []

    total_loss = 0.0

    with torch.no_grad():

        for batch in data_loader:

            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels = batch[
                "labels"
            ].to(device)

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

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    restaurant_path = (
        project_root
        / "data"
        / "restaurant"
        / "train-00000-of-00001.parquet"
    )

    laptop_model_path = (
        project_root
        / "models"
        / "laptop_group_split"
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
    print("Cross-Domain ABSA Experiment")
    print("Train Domain: Laptop")
    print("Target Domain: Restaurant")
    print("=" * 60)

    print(f"\nDevice: {device}")

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Restaurant data
    # --------------------------------------------------------

    print("\nLoading Restaurant data...")

    restaurant_df = load_data(
        restaurant_path
    )

    print(
        f"Restaurant labeled samples: "
        f"{len(restaurant_df)}"
    )

    restaurant_test = (
        create_restaurant_test_split(
            restaurant_df
        )
    )

    print(
        f"Restaurant held-out test samples: "
        f"{len(restaurant_test)}"
    )

    print("\nTest label distribution:")

    print(
        restaurant_test[
            "label"
        ].value_counts()
    )

    # --------------------------------------------------------
    # Load Laptop tokenizer/model
    # --------------------------------------------------------

    print(
        "\nLoading Laptop-trained model..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            laptop_model_path
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            laptop_model_path
        )
    )

    model.to(device)

    print(
        f"Model loaded from:\n"
        f"{laptop_model_path}"
    )

    # --------------------------------------------------------
    # Restaurant test loader
    # --------------------------------------------------------

    test_dataset = ABSADataset(
        restaurant_test,
        tokenizer,
    )

    test_loader = (
        torch.utils.data.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
    )

    # --------------------------------------------------------
    # Cross-domain evaluation
    # --------------------------------------------------------

    print(
        "\n===== Cross-Domain Evaluation ====="
    )

    (
        test_loss,
        accuracy,
        macro_f1,
        true_labels,
        predictions,
    ) = evaluate(
        model,
        test_loader,
        device,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CROSS-DOMAIN RESULTS")
    print("=" * 60)

    print("Source domain: Laptop")
    print("Target domain: Restaurant")

    print(
        f"Test samples: "
        f"{len(restaurant_test)}"
    )

    print(
        f"Accuracy: "
        f"{accuracy:.2%}"
    )

    print(
        f"Macro F1: "
        f"{macro_f1:.4f}"
    )

    print(
        f"Test Loss: "
        f"{test_loss:.4f}"
    )

    print(
        "\nClassification Report:"
    )

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

    # --------------------------------------------------------
    # Compare with Laptop in-domain baseline
    # --------------------------------------------------------

    laptop_accuracy = 0.7358
    laptop_macro_f1 = 0.6465

    print("=" * 60)
    print("DOMAIN COMPARISON")
    print("=" * 60)

    print(
        f"Laptop -> Laptop Accuracy: "
        f"{laptop_accuracy:.2%}"
    )

    print(
        f"Laptop -> Restaurant Accuracy: "
        f"{accuracy:.2%}"
    )

    print()

    print(
        f"Laptop -> Laptop Macro F1: "
        f"{laptop_macro_f1:.4f}"
    )

    print(
        f"Laptop -> Restaurant Macro F1: "
        f"{macro_f1:.4f}"
    )

    accuracy_change = (
        accuracy - laptop_accuracy
    )

    f1_change = (
        macro_f1 - laptop_macro_f1
    )

    print()

    print(
        f"Accuracy change: "
        f"{accuracy_change:+.2%}"
    )

    print(
        f"Macro F1 change: "
        f"{f1_change:+.4f}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()