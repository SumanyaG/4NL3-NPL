from datasets import load_dataset
from llama_cpp import Llama
import time
import re
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
import json

MODEL_PATH = "/Users/sg/Desktop/courses/winter-2025/4nl3/assignments/assignment-4/llama_models/llama-2-7b.Q4_K_M.gguf"
TEST_SPLIT_SIZE = 1000

EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement",
    "fear", "gratitude", "grief", "joy", "love",
    "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

def initialize_model():
    print("⏳ Loading model...")
    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_gpu_layers=1,
            verbose=False
        )
        print("Model loaded successfully")
        return llm
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit(1)

def classify_emotion(llm, text):
    prompt = f"""You are an emotion classifier. Analyze the text and respond ONLY with the corresponding emotion numbers (0-27) between square brackets.
    
    Available emotions:
    0=admiration 1=amusement 2=anger 3=annoyance 4=approval
    5=caring 6=confusion 7=curiosity 8=desire 9=disappointment
    10=disapproval 11=disgust 12=embarrassment 13=excitement
    14=fear 15=gratitude 16=grief 17=joy 18=love
    19=nervousness 20=optimism 21=pride 22=realization
    23=relief 24=remorse 25=sadness 26=surprise 27=neutral

    Text: "{text}"
Answer: ["""
    
    try:
        start_time = time.time()
        response = llm(
            prompt=prompt,
            max_tokens=10,
            temperature=0.1,
            stop=["\n", "]", ","],
            top_p=0.9
        )
        
        raw_output = response['choices'][0]['text'].strip()
        
        numbers = []
        matches = re.findall(r'\b\d+\b', raw_output)
        if matches:
            numbers = [int(num) for num in matches if 0 <= int(num) <= 27]
        
        if not numbers:
            numbers = []
        
        elapsed = time.time() - start_time
        return numbers, elapsed
        
    except Exception as e:
        print(f"Error during classification: {e}")
        return [], 0  

def compute_metrics(true_labels, pred_labels):
    true_binary = np.zeros((len(true_labels), len(EMOTIONS)))
    pred_binary = np.zeros((len(pred_labels), len(EMOTIONS)))
    
    for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
        for label in true:
            true_binary[i][label] = 1
        for label in pred:
            pred_binary[i][label] = 1
    
    micro_f1 = f1_score(true_binary, pred_binary, average='micro', zero_division=0)
    macro_f1 = f1_score(true_binary, pred_binary, average='macro', zero_division=0)
    weighted_f1 = f1_score(true_binary, pred_binary, average='weighted', zero_division=0)
    
    exact_match = np.all(true_binary == pred_binary, axis=1).mean()

    any_match = np.any((true_binary == 1) & (pred_binary == 1), axis=1).mean()
    
    return {
        "exact_match_accuracy": exact_match,
        "any_match_accuracy": any_match,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }

def evaluate_prompt(llm, dataset):
    true_labels = []
    pred_labels = []
    processing_times = []
    confusion_matrix = [[0] * len(EMOTIONS) for _ in range(len(EMOTIONS))]
    
    print(f"\nEvaluating on {len(dataset)} test samples...")
    
    for example in tqdm(dataset, desc="Processing"):
        text = example["text"]
        true_indices = np.where(example["labels"])[0].tolist()
        true_labels.append(true_indices)

        pred_indices, elapsed = classify_emotion(llm, text)
        pred_labels.append(pred_indices)
        processing_times.append(elapsed)
        
        for true_idx in true_indices:
            for pred_idx in pred_indices:
                confusion_matrix[true_idx][pred_idx] += 1
    
    metrics = compute_metrics(true_labels, pred_labels)
    avg_processing_time = np.mean(processing_times)
    
    return metrics, confusion_matrix, avg_processing_time

if __name__ == "__main__":
    print("Loading GoEmotions dataset...")
    dataset = load_dataset("go_emotions", split="test")
    dataset = dataset.select(range(TEST_SPLIT_SIZE)) 
    
    llm = initialize_model()
    
    metrics, cm, avg_time = evaluate_prompt(llm, dataset)
    
    print("\nEvaluation Results:")
    print(f"Average processing time: {avg_time:.2f}s per sample")
    print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.2%}")
    print(f"Any Match Accuracy: {metrics['any_match_accuracy']:.2%}")
    print(f"Micro F1: {metrics['micro_f1']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    
    print("\nConfusion Matrix (True x Predicted):")
    print(f"{'':<15}", end="")
    for i in range(len(EMOTIONS)):
        print(f"{i:<3}", end=" ")
    print()
    
    for i, emotion in enumerate(EMOTIONS):
        print(f"{i:<2}{emotion[:12]:<13}", end="")
        for j in range(len(EMOTIONS)):
            print(f"{cm[i][j]:<3}", end=" ")
        print()
    
    results = {
        "metrics": metrics,
        "average_processing_time": avg_time,
        "confusion_matrix": cm
    }
    
    with open("llama_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nEvaluation complete. Results saved to 'llama_evaluation_results.json'")