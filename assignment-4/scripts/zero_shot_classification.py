from llama_cpp import Llama
import time
import re

MODEL_PATH = "/Users/sg/Desktop/courses/winter-2025/4nl3/assignments/assignment-4/llama_models/deepseek-llm-7b-chat.Q4_K_M.gguf"

EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement",
    "fear", "gratitude", "grief", "joy", "love",
    "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

def initialize_model():
    print("Loading model...")
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
        elapsed = time.time() - start_time
        
        numbers = []
        matches = re.findall(r'\b\d+\b', raw_output)
        if matches:
            numbers = [int(num) for num in matches if 0 <= int(num) <= 27]
        
        if not numbers:
            numbers = []  
            pred_emotions = [""]
        else:
            pred_emotions = [EMOTIONS[num] for num in numbers]
        
        return numbers, pred_emotions, elapsed
        
    except Exception as e:
        print(f"Error during classification: {e}")
        return [], [""], 0  

if __name__ == "__main__":
    llm = initialize_model()
    
    comments = [
        {
            "text": "My favourite food is anything I didn't have to cook myself",
            "true_labels": [27]  
        },
        {
            "text": "Now if he does off himself, everyone will think hes having a laugh screwing with people instead of actually dead",
            "true_labels": [27]  
        },
        {
            "text": "WHY THE FUCK IS BAYLESS ISOING",
            "true_labels": [2]  
        },
        {
            "text": "To make her feel threatened",
            "true_labels": [14] 
        },
        {
            "text": "Dirty Southern Wankers",
            "true_labels": [3] 
        }
    ]
    
    print("\n" + "="*70)
    print(f"{'TEXT':<40} | {'TRUE':<20} | {'PREDICTED':<20} | TIME")
    print("="*70)
    
    for comment in comments:
        text = comment["text"]
        true_indices = comment["true_labels"]
        true_emotions = [EMOTIONS[i] for i in true_indices]
        
        pred_indices, pred_emotions, elapsed = classify_emotion(llm, text)

        text_short = (text[:37] + '...') if len(text) > 40 else text
        true_str = f"{true_indices} → {true_emotions}"
        pred_str = f"{pred_indices} → {pred_emotions}"
        
        print(f"{text_short:<40} | {true_str:<20} | {pred_str:<20} | {elapsed:.2f}s")
        print("-"*70)
    
    print("\nClassification complete")