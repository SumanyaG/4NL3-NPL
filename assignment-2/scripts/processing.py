import os
import re
import numpy as np
import pandas as pd
import nltk
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
from gensim import corpora
import pyLDAvis.gensim_models
import warnings

warnings.filterwarnings('ignore')

def load_texts_from_directory(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                texts.append(f.read())
    return texts

def clean_text(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'(\w+)-\s*(\w+)', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\d+', '', text)  
    text = re.sub(r'[^\w\s]', '', text)  
    
    return text.strip().lower()

def preprocess_text_for_naive_bayes(texts):
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

def preprocess_text_for_lda(texts):
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    additional_stops = {
        'said', 'per', 'one', 'would', 'could', 
        'sgpohearings', 'sgpohearingstxt', 
        'nov', 'dec', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 
        'jul', 'aug', 'sep', 'oct',
        'txt', 'sgpo'
    }
    stop_words.update(additional_stops)
    
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
                token not in stop_words and
                not token.startswith('sgpo'))  
        ]
        
        if tokens: 
            processed_texts.append(tokens)
    
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

def create_lda_model(processed_texts, num_topics=20, passes=25):
    dictionary = corpora.Dictionary(processed_texts)
    
    dictionary.filter_extremes(no_below=2, no_above=0.3)
    
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    
    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=20,
        random_state=42,
        passes=25,
        alpha='auto',
        per_word_topics=True
    )
    
    return lda_model, corpus, dictionary 

def get_topic_words(lda_model, num_words=25):
    topics = []
    for topic_id in range(lda_model.num_topics):
        topic_words = lda_model.show_topic(topic_id, num_words)
        topics.append(topic_words)
    return topics

def get_document_topics(lda_model, corpus, labels):
    doc_topics = [lda_model.get_document_topics(doc) for doc in corpus]
    
    dense_topics = np.zeros((len(doc_topics), lda_model.num_topics))
    for i, doc in enumerate(doc_topics):
        for topic_id, prob in doc:
            dense_topics[i, topic_id] = prob
    
    female_avg = dense_topics[labels == 0].mean(axis=0)
    male_avg = dense_topics[labels == 1].mean(axis=0)
    
    return female_avg, male_avg

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
    
    female_texts = load_texts_from_directory(female_dir)
    male_texts = load_texts_from_directory(male_dir)
    
    all_texts = female_texts + male_texts
    labels = np.array([0]*len(female_texts) + [1]*len(male_texts))
    
    nb_texts = preprocess_text_for_naive_bayes(all_texts)
    
    bow_matrix, vectorizer = create_bow_matrix(nb_texts)
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    p_w_female, p_w_male = calculate_word_probabilities(
        bow_matrix, labels, len(feature_names)
    )
    
    llr_scores = calculate_log_likelihood_ratio(p_w_female, p_w_male)
    
    categorized_results = categorize_words(llr_scores, feature_names)
    
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
    print("\nNaive Bayes results saved to 'naive_bayes_results.csv'")
    
    processed_texts = preprocess_text_for_lda(all_texts)
    
    lda_model, corpus, dictionary = create_lda_model(processed_texts, num_topics=10)
    
    topics = get_topic_words(lda_model)
    
    topic_df = pd.DataFrame(columns=['Topic Label', 'Top Words'])
    for idx, topic in enumerate(topics):
        words_probs = [f"{word} ({prob:.3f})" for word, prob in topic]
        topic_df.loc[idx] = [f"Topic {idx}", ", ".join(words_probs)]
    
    topic_df.to_csv('topic_words.csv', index=False)
    print("\nTop words for each topic saved to 'topic_words.csv'")
    
    female_avg, male_avg = get_document_topics(lda_model, corpus, labels)
    
    category_topics = pd.DataFrame({
        'Topic': range(lda_model.num_topics),
        'Female Average': female_avg,
        'Male Average': male_avg
    })
    category_topics = category_topics.sort_values('Female Average', ascending=False)
    category_topics.to_csv('category_topics.csv', index=False)
    
    print("\nTop 3 topics for Female category:")
    for idx in female_avg.argsort()[-3:][::-1]:
        print(f"Topic {idx} ({female_avg[idx]:.3f})")
        print("Top words:", ", ".join([f"{w}({p:.3f})" for w, p in topics[idx][:5]]))
    
    print("\nTop 3 topics for Male category:")
    for idx in male_avg.argsort()[-3:][::-1]:
        print(f"Topic {idx} ({male_avg[idx]:.3f})")
        print("Top words:", ", ".join([f"{w}({p:.3f})" for w, p in topics[idx][:5]]))
    
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, 'lda_visualization.html')
    print("Interactive visualization saved to 'lda_visualization.html'")

if __name__ == "__main__":
    main()