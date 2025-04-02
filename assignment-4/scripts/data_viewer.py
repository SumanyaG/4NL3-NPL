from datasets import load_dataset
from collections import Counter
import pandas as pd

dataset = load_dataset("go_emotions", "simplified")
label_names = dataset['train'].features['labels'].feature.names

def label_mapping(label_names):
    for idx, label in enumerate(label_names):
        print(f"{idx}: {label}")

def count_labels(df):
    label_counts = Counter()
    for labels in df["labels"]:
        for label in labels: 
            label_counts[label] += 1
    return label_counts

def class_distribution(label_names):
    train_df = dataset['train'].to_pandas()
    test_df = dataset['test'].to_pandas()

    train_label_counts = count_labels(train_df)
    test_label_counts = count_labels(test_df)

    label_stats = pd.DataFrame({
    "Emotion": label_names,
    "Train Count": [train_label_counts[i] for i in range(len(label_names))],
    "Test Count": [test_label_counts[i] for i in range(len(label_names))]
    })

    return label_stats

if __name__ == "__main__":
    print(class_distribution(label_names))
