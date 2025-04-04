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

import matplotlib.pyplot as plt
import pandas as pd

def visualize_class_distribution(label_stats):
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)

    bar_width = 0.35
    x = range(len(label_stats))
    
    train_bars = ax.bar(x, label_stats["Train Count"], width=bar_width, 
                        label='Train', color='royalblue', alpha=0.7)
    test_bars = ax.bar([p + bar_width for p in x], label_stats["Test Count"], 
                       width=bar_width, label='Test', color='orange', alpha=0.7)

    ax.set_xlabel('Emotion Class', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Class Distribution in Train and Test Sets', fontsize=14, pad=20)
    ax.set_xticks([p + bar_width/2 for p in x])
    ax.set_xticklabels(label_stats["Emotion"], rotation=45, ha='right')
    ax.legend()
    
    plt.show()

if __name__ == "__main__":
    label_stats = class_distribution(label_names)
    visualize_class_distribution(label_stats)
