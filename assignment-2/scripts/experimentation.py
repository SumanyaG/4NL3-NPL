import processing

import pyLDAvis
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer

def preprocess_text_with_stemming(texts):
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    additional_stops = {
        'said', 'would', 'could', 'also', 'one', 'two', 'three',
        'hearing', 'hearings', 'committee', 'senate', 'senator',
        'court', 'supreme', 'judicial', 'judge', 'case', 'law',
        'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
        'sgpohearings', 'sgpohearingstxt', 'txt', 'sgpo'
    }
    stop_words.update(additional_stops)
    
    processed_texts = []
    for text in texts:
        text = processing.clean_text(text)
        words = [stemmer.stem(word) for word in text.split() 
                if len(word) > 2 
                and word not in stop_words
                and not any(char.isdigit() for char in word)
                and not word.startswith('sgpo')]
        processed_texts.append(' '.join(words))
    
    return processed_texts

def create_tfidf_matrix(texts):
    vectorizer = TfidfVectorizer(
        min_df=5,
        max_df=0.6,
        token_pattern=r'[a-zA-Z]+(?:[-][a-zA-Z]+)*',
        lowercase=True
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer

def main():
    female_dir = "/Users/sg/Desktop/courses/winter-2025/4nl3/assignments/assignment-2/dataset/female-nominees-processed"
    male_dir = "/Users/sg/Desktop/courses/winter-2025/4nl3/assignments/assignment-2/dataset/male-nominees-processed"
    
    female_texts = processing.load_texts_from_directory(female_dir)
    male_texts = processing.load_texts_from_directory(male_dir)
    all_texts = female_texts + male_texts
    labels = np.array([0]*len(female_texts) + [1]*len(male_texts))
    
    stemmed_texts = preprocess_text_with_stemming(all_texts)
    
    tfidf_matrix, vectorizer = create_tfidf_matrix(stemmed_texts)
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    p_w_female, p_w_male = processing.calculate_word_probabilities(
        tfidf_matrix, labels, len(feature_names)
    )
    
    llr_scores = processing.calculate_log_likelihood_ratio(p_w_female, p_w_male)
    categorized_results = processing.categorize_words(llr_scores, feature_names)
    
    lda_texts = [text.split() for text in stemmed_texts]
    lda_model, corpus, dictionary = processing.create_lda_model(lda_texts, num_topics=10)
    
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
    
    pd.DataFrame(results_list).to_csv('stemmed_tfidf_results.csv', index=False)
    
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, 'stemmed_lda_visualization.html')

if __name__ == "__main__":
    main()