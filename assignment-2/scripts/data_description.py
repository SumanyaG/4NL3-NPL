import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from collections import defaultdict

def tokenize_text(text):
    text = text[1:] if text.startswith('\ufeff') else text
    pattern = r"[A-Za-z]+(?:[''][A-Za-z]+)*|\d+|[A-Za-z]+|[^\w\s]"
    tokens = re.findall(pattern, text)
    tokens = [token for token in tokens if any(c.isalnum() for c in token)]
    return tokens

def process_directory(directory):
    file_stats = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            tokens = tokenize_text(text)
            
            stats = {
                'filename': filename,
                'total_tokens': len(tokens),
                'unique_tokens': len(set(tokens)),
                'token_frequency': defaultdict(int)
            }
            
            for token in tokens:
                stats['token_frequency'][token] += 1
            
            file_stats.append(stats)
    
    return file_stats

def analyze_categories(female_dir, male_dir):
    female_stats = process_directory(female_dir)
    male_stats = process_directory(male_dir)
    
    female_avg_tokens = sum(stat['total_tokens'] for stat in female_stats) / len(female_stats)
    female_avg_unique = sum(stat['unique_tokens'] for stat in female_stats) / len(female_stats)
    
    male_avg_tokens = sum(stat['total_tokens'] for stat in male_stats) / len(male_stats)
    male_avg_unique = sum(stat['unique_tokens'] for stat in male_stats) / len(male_stats)
    
    female_total_freq = defaultdict(int)
    for stat in female_stats:
        for token, freq in stat['token_frequency'].items():
            female_total_freq[token] += freq
            
    male_total_freq = defaultdict(int)
    for stat in male_stats:
        for token, freq in stat['token_frequency'].items():
            male_total_freq[token] += freq
    
    return {
        'female': {
            'files': [stat['filename'] for stat in female_stats],
            'individual_stats': female_stats,
            'avg_tokens': female_avg_tokens,
            'avg_unique_tokens': female_avg_unique,
            'total_frequency': female_total_freq
        },
        'male': {
            'files': [stat['filename'] for stat in male_stats],
            'individual_stats': male_stats,
            'avg_tokens': male_avg_tokens,
            'avg_unique_tokens': male_avg_unique,
            'total_frequency': male_total_freq
        }
    }

def visualize_results(results, output_dir='visualizations'):
    os.makedirs(output_dir, exist_ok=True)
    
    avg_stats = pd.DataFrame({
        'Category': ['Female Nominees', 'Male Nominees'],
        'Average Tokens': [results['female']['avg_tokens'], results['male']['avg_tokens']],
        'Average Unique Tokens': [results['female']['avg_unique_tokens'], results['male']['avg_unique_tokens']]
    })
    
    avg_stats.to_csv(os.path.join(output_dir, 'average_statistics.csv'), index=False)
    
    plt.figure(figsize=(10, 6))
    x = range(len(avg_stats['Category']))
    width = 0.35
    
    plt.bar(x, avg_stats['Average Tokens'], width, label='Total Tokens')
    plt.bar([i + width for i in x], avg_stats['Average Unique Tokens'], width, label='Unique Tokens')
    
    plt.xlabel('Category')
    plt.ylabel('Number of Tokens')
    plt.title('Average Token Statistics by Category')
    plt.xticks([i + width/2 for i in x], avg_stats['Category'])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'token_statistics.png'))
    plt.close()

    female_stats = pd.DataFrame([(stat['filename'], stat['total_tokens'], stat['unique_tokens']) 
                               for stat in results['female']['individual_stats']],
                              columns=['Filename', 'Total Tokens', 'Unique Tokens'])
    male_stats = pd.DataFrame([(stat['filename'], stat['total_tokens'], stat['unique_tokens']) 
                             for stat in results['male']['individual_stats']],
                            columns=['Filename', 'Total Tokens', 'Unique Tokens'])
    
    female_stats.to_csv(os.path.join(output_dir, 'female_nominee_statistics.csv'), index=False)
    male_stats.to_csv(os.path.join(output_dir, 'male_nominee_statistics.csv'), index=False)

def main():
    female_dir = "/Users/sg/Desktop/courses/winter-2025/4nl3/assignments/assignment-2/dataset/female-nominees-processed"
    male_dir = "/Users/sg/Desktop/courses/winter-2025/4nl3/assignments/assignment-2/dataset/male-nominees-processed"
    
    results = analyze_categories(female_dir, male_dir)
    
    print("\nFemale Nominees:")
    print(f"Number of files analyzed: {len(results['female']['files'])}")
    print(f"Average tokens per file: {results['female']['avg_tokens']:.2f}")
    print(f"Average unique tokens per file: {results['female']['avg_unique_tokens']:.2f}")
    
    print("\nMale Nominees:")
    print(f"Number of files analyzed: {len(results['male']['files'])}")
    print(f"Average tokens per file: {results['male']['avg_tokens']:.2f}")
    print(f"Average unique tokens per file: {results['male']['avg_unique_tokens']:.2f}")

    visualize_results(results)
    print("Analysis complete.")

if __name__ == "__main__":
    main()