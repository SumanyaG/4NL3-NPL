import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import re
import pandas as pd
from gensim import corpora
import gensim
import logging
from scipy.sparse import save_npz, load_npz
import math
from collections import defaultdict

class TextNormalizer:
    def __init__(self, lowercase=True, stem=False, lemmatize=False, 
                 remove_stopwords=True, remove_numbers=True):
        self.lowercase = lowercase
        self.stem = stem
        self.lemmatize = lemmatize
        self.remove_stopwords = remove_stopwords
        self.remove_numbers = remove_numbers
        
        # Initialize NLTK components
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
        
        self.stemmer = PorterStemmer() if stem else None
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
    
    def normalize(self, text):
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Process tokens
        normalized_tokens = []
        for token in tokens:
            # Skip stopwords
            if self.remove_stopwords and token in self.stop_words:
                continue
            
            # Apply stemming or lemmatization
            if self.stem:
                token = self.stemmer.stem(token)
            elif self.lemmatize:
                token = self.lemmatizer.lemmatize(token)
            
            normalized_tokens.append(token)
        
        return ' '.join(normalized_tokens)

def create_bow_representations(texts, vectorizer_type='count'):
    """Create different BOW representations"""
    if vectorizer_type == 'count':
        vectorizer = CountVectorizer(min_df=2, max_df=0.95)
    elif vectorizer_type == 'binary':
        vectorizer = CountVectorizer(min_df=2, max_df=0.95, binary=True)
    else:  # tfidf
        vectorizer = TfidfVectorizer(min_df=2, max_df=0.95)
    
    bow_matrix = vectorizer.fit_transform(texts)
    return bow_matrix, vectorizer

def calculate_llr(bow_matrix, labels, vectorizer):
    """Calculate log likelihood ratios for words"""
    female_docs = bow_matrix[labels == 0]
    male_docs = bow_matrix[labels == 1]
    
    # Get word counts
    female_word_counts = np.array(female_docs.sum(axis=0))[0]
    male_word_counts = np.array(male_docs.sum(axis=0))[0]
    
    # Add smoothing
    alpha = 1.0
    vocab_size = len(vectorizer.vocabulary_)
    
    # Calculate probabilities with smoothing
    p_w_female = (female_word_counts + alpha) / (female_word_counts.sum() + alpha * vocab_size)
    p_w_male = (male_word_counts + alpha) / (male_word_counts.sum() + alpha * vocab_size)
    
    # Calculate LLR
    llr = np.log(p_w_female) - np.log(p_w_male)
    
    return llr

def run_lda(texts, num_topics=10):
    """Run LDA and return coherence score"""
    # Tokenize
    tokenized_texts = [text.split() for text in texts]
    
    # Create dictionary and corpus
    dictionary = corpora.Dictionary(tokenized_texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    
    # Train LDA
    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=15
    )
    
    # Calculate coherence
    coherence_model = gensim.models.CoherenceModel(
        model=lda_model, 
        texts=tokenized_texts, 
        dictionary=dictionary, 
        coherence='c_v'
    )
    
    return coherence_model.get_coherence()

def compare_configurations(female_texts, male_texts):
    """Compare different configurations and their results"""
    results = []
    
    # Define configurations to test
    normalizer_configs = [
        {'name': 'basic', 'lowercase': True, 'remove_stopwords': True, 'remove_numbers': True},
        {'name': 'stem', 'lowercase': True, 'stem': True, 'remove_stopwords': True, 'remove_numbers': True},
        {'name': 'lemmatize', 'lowercase': True, 'lemmatize': True, 'remove_stopwords': True, 'remove_numbers': True}
    ]
    
    bow_types = ['count', 'binary', 'tfidf']
    
    for norm_config in normalizer_configs:
        normalizer = TextNormalizer(**{k: v for k, v in norm_config.items() if k != 'name'})
        
        # Normalize texts
        normalized_female = [normalizer.normalize(text) for text in female_texts]
        normalized_male = [normalizer.normalize(text) for text in male_texts]
        all_texts = normalized_female + normalized_male
        labels = np.array([0]*len(normalized_female) + [1]*len(normalized_male))
        
        for bow_type in bow_types:
            print(f"\nProcessing {norm_config['name']} normalization with {bow_type} representation...")
            
            # Create BOW representation
            bow_matrix, vectorizer = create_bow_representations(all_texts, bow_type)
            
            # Calculate LLR
            llr = calculate_llr(bow_matrix, labels, vectorizer)
            
            # Get top words by LLR
            idx_to_word = {idx: word for word, idx in vectorizer.vocabulary_.items()}
            top_female = [(idx_to_word[i], llr[i]) for i in llr.argsort()[-10:]]
            top_male = [(idx_to_word[i], llr[i]) for i in llr.argsort()[:10]]
            
            # Run LDA
            coherence = run_lda(all_texts)
            
            # Store results
            results.append({
                'normalization': norm_config['name'],
                'bow_type': bow_type,
                'vocab_size': len(vectorizer.vocabulary_),
                'coherence': coherence,
                'top_female_words': top_female,
                'top_male_words': top_male
            })
    
    return pd.DataFrame(results)

def main():
    # Load your corpus
    print("Loading corpus...")
    with open('female_texts.txt', 'r', encoding='utf-8') as f:
        female_texts = f.readlines()
    with open('male_texts.txt', 'r', encoding='utf-8') as f:
        male_texts = f.readlines()
    
    # Run comparisons
    results_df = compare_configurations(female_texts, male_texts)
    
    # Save results
    results_df.to_csv('normalization_comparison.csv', index=False)
    
    # Print summary
    print("\nSummary of results:")
    print("\nBest configurations by coherence score:")
    print(results_df.sort_values('coherence', ascending=False)[['normalization', 'bow_type', 'coherence']].head())
    
    print("\nVocabulary sizes:")
    print(results_df.groupby(['normalization', 'bow_type'])['vocab_size'].mean())
    
    # Create detailed report
    with open('analysis_report.txt', 'w') as f:
        f.write("Text Normalization and BOW Comparison Analysis\n")
        f.write("===========================================\n\n")
        
        for _, row in results_df.iterrows():
            f.write(f"\nConfiguration: {row['normalization']} normalization with {row['bow_type']} representation\n")
            f.write(f"Vocabulary size: {row['vocab_size']}\n")
            f.write(f"Topic coherence: {row['coherence']:.4f}\n")
            f.write("\nTop female-associated words:\n")
            for word, score in row['top_female_words']:
                f.write(f"  {word}: {score:.4f}\n")
            f.write("\nTop male-associated words:\n")
            for word, score in row['top_male_words']:
                f.write(f"  {word}: {score:.4f}\n")
            f.write("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()