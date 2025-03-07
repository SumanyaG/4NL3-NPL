import os
import re
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec, KeyedVectors
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

from datasets import load_dataset
dataset = load_dataset("wikipedia", "20220301.simple")

def clean_text(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'(\w+)-\s*(\w+)', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()

def preprocess_text(texts):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    processed_texts = []
    for text in texts:
        cleaned_text = clean_text(text)
        tokens = word_tokenize(cleaned_text.lower())
        tokens = [
            lemmatizer.lemmatize(token) 
            for token in tokens 
            if (token.isalpha() and 
                len(token) > 2 and 
                not any(char.isdigit() for char in token) and 
                token not in stop_words)]
        
        if tokens:
            processed_texts.append(tokens)
    
    return processed_texts

def train_word_embeddings(corpus, save_path="models/"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print("Training Word2Vec models...")
    
    # Model 1: Skip-gram with window size 5, 100 dimensions
    model_sg = Word2Vec(
        sentences=corpus,
        vector_size=100,
        window=5,
        min_count=5, 
        workers=4,
        sg=1
    )
    
    # Model 2: CBOW with window size 10, 200 dimensions
    model_cbow = Word2Vec(
        sentences=corpus,
        vector_size=200,
        window=10,
        min_count=5,
        workers=4,
        sg=0 
    )
    
    # Model 3: Skip-gram with larger vocabulary (lower min_count)
    model_large_vocab = Word2Vec(
        sentences=corpus,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        sg=1
    )
    
    model_sg.save(os.path.join(save_path, "word2vec_skipgram.model"))
    model_cbow.save(os.path.join(save_path, "word2vec_cbow.model"))
    model_large_vocab.save(os.path.join(save_path, "word2vec_large_vocab.model"))
    
    vocab_size_sg = len(model_sg.wv.index_to_key)
    vocab_size_cbow = len(model_cbow.wv.index_to_key)
    vocab_size_large = len(model_large_vocab.wv.index_to_key)
    
    print(f"Model 1 (Skip-gram) vocabulary size: {vocab_size_sg}")
    print(f"Model 2 (CBOW) vocabulary size: {vocab_size_cbow}")
    print(f"Model 3 (Large vocab) vocabulary size: {vocab_size_large}")
    
    print(f"Vocabulary size increase in Model 3 vs Model 1: {vocab_size_large - vocab_size_sg} words ({(vocab_size_large/vocab_size_sg - 1)*100:.2f}%)")
    
    return {
        "skipgram": model_sg,
        "cbow": model_cbow,
        "large_vocab": model_large_vocab
    }

def load_pretrained_embeddings(save_path="models/"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    pretrained_models = {}

    glove_path = os.path.join(save_path, "glove.6B.100d.txt")
    from gensim.scripts.glove2word2vec import glove2word2vec
    
    glove_word2vec_path = os.path.join(save_path, "glove.6B.100d.word2vec.txt")
    if not os.path.exists(glove_word2vec_path):
        glove2word2vec(glove_path, glove_word2vec_path)
    
    pretrained_models["glove"] = KeyedVectors.load_word2vec_format(glove_word2vec_path)
    
    word2vec_path = os.path.join(save_path, "GoogleNews-vectors-negative300.bin")
    pretrained_models["word2vec_google"] = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    
    return pretrained_models

def create_classification_dataset():
    print("Creating classification dataset...")
    articles = dataset["train"]["text"][:1000]

    texts = [article for article in articles if len(article.split()) > 20]
    financial_keywords = ['market', 'stock', 'economy', 'financial', 'business', 
                         'investment', 'economic', 'trade', 'bank', 'finance']
    
    labels = []
    for text in texts:
        text_lower = text.lower()
        is_financial = any(keyword in text_lower for keyword in financial_keywords)
        labels.append(1 if is_financial else 0)

    num_financial = sum(labels)
    print(f"Created dataset with {len(texts)} documents")
    print(f"Financial articles: {num_financial}, Non-financial: {len(texts) - num_financial}")
    
    return texts, labels

def text_to_embedding(text, model, tokenize=True):
    if tokenize:
        cleaned_text = clean_text(text)
        tokens = word_tokenize(cleaned_text.lower())
    else:
        tokens = text  

    if hasattr(model, 'wv'): 
        vectors = [model.wv[word] for word in tokens if word in model.wv]
    else:  
        vectors = [model[word] for word in tokens if word in model]
    
    if not vectors: 
        if hasattr(model, 'wv'):
            vector_size = model.vector_size
        else:
            vector_size = model.vector_size
        return np.zeros(vector_size)
    
    return np.mean(vectors, axis=0)

def train_and_evaluate_classifier(X_train, X_test, y_train, y_test, feature_type, model=None, model_name=None):
    if feature_type == 'bow':
        vectorizer = CountVectorizer(min_df=2, max_df=0.7)
        X_train_features = vectorizer.fit_transform(X_train)
        X_test_features = vectorizer.transform(X_test)
        feature_dim = X_train_features.shape[1]
    
    elif feature_type == 'embedding':
        X_train_features = np.array([text_to_embedding(text, model) for text in X_train])
        X_test_features = np.array([text_to_embedding(text, model) for text in X_test])
        feature_dim = X_train_features.shape[1]
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_features, y_train)
    
    y_pred = clf.predict(X_test_features)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n{model_name} Model Results:")
    print(f"Feature dimension: {feature_dim}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy, f1

def analyze_vocabulary_coverage(models, texts):
    print("\nAnalyzing vocabulary coverage...")
    
    all_tokens = []
    for text in texts:
        cleaned_text = clean_text(text)
        tokens = word_tokenize(cleaned_text.lower())
        all_tokens.extend(tokens)
    
    unique_tokens = set(all_tokens)
    print(f"Total unique tokens in corpus: {len(unique_tokens)}")
    
    coverage_stats = {}
    
    for model_name, model in models.items():
        if hasattr(model, 'wv'):
            vocab = set(model.wv.index_to_key)
            vocab_size = len(vocab)
        else:
            vocab = set(model.index_to_key)
            vocab_size = len(vocab)
        
        covered_tokens = unique_tokens.intersection(vocab)
        coverage_percentage = len(covered_tokens) / len(unique_tokens) * 100
        
        coverage_stats[model_name] = {
            'vocab_size': vocab_size,
            'covered_tokens': len(covered_tokens),
            'coverage_percentage': coverage_percentage
        }
        
        print(f"{model_name.capitalize()} model:")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Tokens covered: {len(covered_tokens)} out of {len(unique_tokens)}")
        print(f"  Coverage: {coverage_percentage:.2f}%")
    
    return coverage_stats

def plot_performance_comparison(results):
    model_names = list(results.keys())
    accuracies = [results[name][0] for name in model_names]
    f1_scores = [results[name][1] for name in model_names]
    
    x = range(len(model_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, accuracies, width, label='Accuracy')
    ax.bar([i + width for i in x], f1_scores, width, label='F1 Score')
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title('Classification Performance Comparison')
    ax.set_xticks([i + width/2 for i in x])
    ax.set_xticklabels(model_names)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

def plot_vocab_coverage(coverage_stats):
    model_names = list(coverage_stats.keys())
    vocab_sizes = [coverage_stats[name]['vocab_size'] for name in model_names]
    coverage_percentages = [coverage_stats[name]['coverage_percentage'] for name in model_names]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Vocabulary Size', color=color)
    ax1.bar(model_names, vocab_sizes, color=color, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Coverage Percentage (%)', color=color)
    ax2.plot(model_names, coverage_percentages, color=color, marker='o', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Vocabulary Size vs. Coverage Percentage')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('vocab_coverage.png')
    plt.show()

def main():
    print("Processing Wikipedia data...")
    sample_size = 10000  # Adjust based on computational resources
    wiki_texts = dataset["train"]["text"][:sample_size]
    
    processed_corpus = preprocess_text(wiki_texts)
    print(f"Processed {len(processed_corpus)} documents")
    
    models = train_word_embeddings(processed_corpus)

    texts, labels = create_classification_dataset()
    
    coverage_stats = analyze_vocabulary_coverage(models, texts)
    plot_vocab_coverage(coverage_stats)
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    results = {}
    
    print("\nTraining Bag of Words classifier...")
    bow_accuracy, bow_f1 = train_and_evaluate_classifier(
        X_train, X_test, y_train, y_test, feature_type='bow', model_name="Bag of Words"
    )
    results["Bag of Words"] = (bow_accuracy, bow_f1)
    
    for model_name, model in models.items():
        print(f"\nTraining classifier with {model_name} embeddings...")
        emb_accuracy, emb_f1 = train_and_evaluate_classifier(
            X_train, X_test, y_train, y_test, 
            feature_type='embedding', 
            model=model, 
            model_name=model_name.capitalize()
        )
        results[model_name.capitalize()] = (emb_accuracy, emb_f1)
    
    plot_performance_comparison(results)
    
    print("\nEfficiency Comparison:")
    vectorizer = CountVectorizer().fit(X_train)
    bow_dim = len(vectorizer.vocabulary_)
    print(f"Bag of Words features dimension: {bow_dim}")
    for model_name, model in models.items():
        if hasattr(model, 'wv'):
            emb_dim = model.vector_size
        else:
            emb_dim = model.wv.vector_size
        print(f"{model_name.capitalize()} Embedding features dimension: {emb_dim}")
        print(f"Dimension reduction ratio: {bow_dim / emb_dim:.2f}x")

if __name__ == "__main__":
    main()