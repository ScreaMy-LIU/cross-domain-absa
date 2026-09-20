import csv
import re
from pathlib import Path


POSITIVE_WORDS = ["good", "great", "excellent", "amazing", "fast"]
NEGATIVE_WORDS = ["bad", "poor", "slow", "terrible", "expensive"]


def get_aspect_context(text, aspect):
    """Find the local clause containing the target aspect."""
    clauses = re.split(r"\bbut\b|\bhowever\b|[,.!?;]", text.lower())

    for clause in clauses:
        if aspect.lower() in clause:
            return clause.strip()

    return None


def predict_sentiment(text, aspect):
    """Predict sentiment for a specific aspect."""
    context = get_aspect_context(text, aspect)

    if context is None:
        return "aspect not found"

    for word in POSITIVE_WORDS:
        if word in context:
            return "positive"

    for word in NEGATIVE_WORDS:
        if word in context:
            return "negative"

    return "neutral"


def evaluate(csv_path):
    """Evaluate the baseline on a CSV dataset."""
    correct = 0
    total = 0

    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            text = row["text"]
            aspect = row["aspect"]
            gold_label = row["label"]

            prediction = predict_sentiment(text, aspect)

            total += 1

            if prediction == gold_label:
                correct += 1

            print(
                f"Aspect: {aspect:12} "
                f"Gold: {gold_label:8} "
                f"Predicted: {prediction}"
            )

    accuracy = correct / total if total > 0 else 0

    print("\n--------------------")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "data" / "sample_absa.csv"

    evaluate(dataset_path)