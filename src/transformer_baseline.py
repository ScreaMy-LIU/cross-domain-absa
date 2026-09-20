import csv
from pathlib import Path

from transformers import pipeline
from sklearn.metrics import accuracy_score


def load_dataset(csv_path):
    """Load the ABSA sample dataset."""
    samples = []

    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            samples.append(row)

    return samples


def main():
    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "data" / "sample_absa.csv"

    # Pre-trained sentiment analysis model
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    samples = load_dataset(dataset_path)

    gold_labels = []
    predictions = []

    for sample in samples:
        text = sample["text"]
        aspect = sample["aspect"]
        gold = sample["label"]

        # Give the model both the review and the target aspect
        model_input = f"{text} Aspect: {aspect}"

        result = classifier(model_input)[0]

        predicted = result["label"].lower()

        gold_labels.append(gold)
        predictions.append(predicted)

        print(
            f"Aspect: {aspect:12} "
            f"Gold: {gold:8} "
            f"Predicted: {predicted:8} "
            f"Score: {result['score']:.4f}"
        )

    # SST-2 only predicts positive/negative.
    # Neutral samples are therefore expected to be problematic.
    accuracy = accuracy_score(gold_labels, predictions)

    print("\n--------------------")
    print(f"Samples: {len(samples)}")
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()