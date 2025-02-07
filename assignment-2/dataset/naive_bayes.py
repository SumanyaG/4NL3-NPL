import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import os
import re
from nltk.corpus import stopwords
import nltk
from scipy.sparse import save_npz
import glob

def read_corpus(female_dir, male_dir):
    texts = []
    labels = [] 
    
    for filepath in glob.glob(os.path.join(female_dir, "*.txt")):
        with open(filepath, 'r', encoding='utf-8') as f:
            texts.append(f.read())
            labels.append(0)

    for filepath in glob.glob(os.path.join(male_dir, "*.txt")):
        with open(filepath, 'r', encoding='utf-8') as f:
            texts.append(f.read())
            labels.append(1)
    
    return texts, labels

def clean_text(text):
    text = re.sub(r'\d+', '', text)  
    text = re.sub(r'[^\w\s]', '', text)  
    text = re.sub(r'\s+', ' ', text) 
    
    return text.strip().lower()

def preprocess_texts(texts):
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    stop_words = set(stopwords.words('english'))
    additional_stops = {
        'said', 'would', 'could', 'also', 'one', 'two', 'three',
        'hearing', 'hearings', 'committee', 'senate', 'senator',
        'court', 'supreme', 'judicial', 'judge', 'case', 'law',
        'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
    }
    stop_words.update(additional_stops)
    
    processed_texts = []
    for text in texts:
        text = clean_text(text)
        words = [word for word in text.split() 
                if len(word) > 2 
                and word not in stop_words
                and not any(char.isdigit() for char in word)]
        processed_texts.append(' '.join(words))
    
    return processed_texts

def create_bow_matrix(texts):
    vectorizer = CountVectorizer(
        min_df=5,  
        max_df=0.6,  
        token_pattern=r'[a-zA-Z]+(?:[-][a-zA-Z]+)*',
        lowercase=True
    )
    
    bow_matrix = vectorizer.fit_transform(texts)
    return bow_matrix, vectorizer

def calculate_word_probabilities(bow_matrix, labels, vocab_size, alpha=1.0):
    female_docs = bow_matrix[labels == 0]
    male_docs = bow_matrix[labels == 1]
    
    female_word_counts = np.array(female_docs.sum(axis=0))[0]
    male_word_counts = np.array(male_docs.sum(axis=0))[0]
    
    total_female_words = female_word_counts.sum() + alpha * vocab_size
    total_male_words = male_word_counts.sum() + alpha * vocab_size
    
    p_w_female = (female_word_counts + alpha) / total_female_words
    p_w_male = (male_word_counts + alpha) / total_male_words
    
    return p_w_female, p_w_male

def calculate_log_likelihood_ratio(p_w_c, p_w_other):
    return np.log(p_w_c) - np.log(p_w_other)

def categorize_words(llr_scores, feature_names, threshold=1.0):
    categories = {
        'legal_interpretation': {
            'constitution', 'precedent', 'interpretation', 'amendment', 'rights',
            'principle', 'doctrine', 'judicial', 'ruling', 'decision'
        },
        'background_experience': {
            'background', 'experience', 'career', 'education', 'history',
            'practice', 'service', 'work', 'professional', 'record'
        },
        'personal_characteristics': {
            'character', 'integrity', 'temperament', 'belief', 'value',
            'identity', 'gender', 'race', 'ethnicity', 'bias'
        },
        'competency_qualification': {
            'qualified', 'competent', 'ability', 'skill', 'knowledge',
            'expertise', 'capability', 'judgment', 'understanding', 'reasoning'
        }
    }
    
    results = {
        'female': {cat: [] for cat in categories.keys()},
        'male': {cat: [] for cat in categories.keys()}
    }
    
    for idx, (word, score) in enumerate(zip(feature_names, llr_scores)):
        for category, category_words in categories.items():
            if any(cat_word in word for cat_word in category_words):
                if score > threshold:
                    results['female'][category].append((word, score))
                elif score < -threshold:
                    results['male'][category].append((word, -score))
    
    return results

def print_results(categorized_words):
    print("\nNaive Bayes Analysis of Supreme Court Nomination Hearings")
    print("=" * 70)
    
    for gender in ['female', 'male']:
        print(f"\n{gender.upper()} NOMINEE HEARINGS")
        print("-" * 40)
        
        for category, words in categorized_words[gender].items():
            if words:
                print(f"\n{category.replace('_', ' ').title()}:")
                sorted_words = sorted(words, key=lambda x: x[1], reverse=True)
                for word, score in sorted_words[:5]:
                    print(f"  {word:<20} (LLR: {score:.3f})")

def main():
    female_dir = "/Users/sg/Desktop/courses/winter-2025/4nl3/assignments/assignment-2/dataset/female-nominees-processed"
    male_dir = "/Users/sg/Desktop/courses/winter-2025/4nl3/assignments/assignment-2/dataset/male-nominees-processed"
    
    print("Loading data...")
    texts, labels = read_corpus(female_dir, male_dir)
    
    print("Preprocessing texts...")
    all_texts = preprocess_texts(texts)
    labels = np.array(labels)
    
    print("Creating BOW matrix...")
    bow_matrix, vectorizer = create_bow_matrix(all_texts)
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    print(f"Vocabulary size: {len(feature_names)}")
    print(f"BOW matrix shape: {bow_matrix.shape}")
    print(f"BOW matrix sparsity: {(1.0 - bow_matrix.nnz / (bow_matrix.shape[0] * bow_matrix.shape[1])) * 100:.2f}%")

    save_npz('bow_matrix.npz', bow_matrix)
    np.save('labels.npy', labels)
    
    print("Calculating Naive Bayes probabilities...")
    p_w_female, p_w_male = calculate_word_probabilities(
        bow_matrix, labels, len(feature_names)
    )
    
    word_freq = [(word, bow_matrix.sum(axis=0)[0, idx]) 
                 for word, idx in vectorizer.vocabulary_.items()]
    top_words = sorted(word_freq, key=lambda x: x[1], reverse=True)[:20]

    print("Calculating log likelihood ratios...")
    llr_scores = calculate_log_likelihood_ratio(p_w_female, p_w_male)
    
    print("Analyzing word categories...")
    categorized_results = categorize_words(llr_scores, feature_names)
    print_results(categorized_results)
    
    results_list = []
    for gender in ['female', 'male']:
        for category, words in categorized_results[gender].items():
            for word, score in words:
                results_list.append({
                    'Gender': gender,
                    'Category': category,
                    'Word': word,
                    'LLR_Score': score
                })
    
    pd.DataFrame(results_list).to_csv('naive_bayes_results.csv', index=False)
    print("\nDetailed results have been saved to 'naive_bayes_results.csv'")

if __name__ == "__main__":
    main()