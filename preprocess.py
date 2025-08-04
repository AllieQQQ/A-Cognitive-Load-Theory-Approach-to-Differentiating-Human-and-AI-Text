import os
import random
import json
import re
from typing import List, Dict
from nltk.tokenize import word_tokenize, sent_tokenize

# clean original texts

# Function to clean text by removing the @ symbols and extra whitespace
def clean_text(text: str) -> str:
    """
    Remove @ symbols (added for copyright), HTML tags, backslashes, 
    and normalize whitespace in the text.
    
    Args:
        text (str): Raw text from the article.
    Returns:
        str: Cleaned text with unwanted symbols removed and whitespace normalized.
    """
    cleaned = re.sub(r'@+', '', text)  # Remove all @ symbols
    cleaned = re.sub(r'<[^>]+>', '', cleaned)  # Remove HTML tags like <p>, <br>, etc.
    cleaned = cleaned.replace('\\', '')  # Remove backslashes
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()  # Normalize multiple spaces to single space
    return cleaned

# Function to split text into words and get first 500 and last 500 words
def split_text_words(text: str) -> tuple:
    """
    Split cleaned text into words and extract the first 500 and last 500 words.
    
    Args:
        text (str): Cleaned article text.
    Returns:
        tuple: (first_500_words, next_500_words) as strings, or empty strings if too short.
    """
    words = text.split()
    if len(words) < 1000:  # Check if text has fewer than 1000 words
        print(f"Warning: Article has only {len(words)} words, less than 1000.")
        return " ".join(words[:500]), ""  # Return first 500 words and empty string if too short
    first_500 = " ".join(words[:500])  # First 500 words
    next_500 = " ".join(words[500:1000])   # Last 500 words
    return first_500, next_500

# Function to process a single article
def process_article(article_id: str, full_text: str) -> Dict:
    """
    Process one article: clean it, extract first 1000 chars, and split into first/last 500 words.
    
    Args:
        article_id (str): 7-digit article ID starting with 400.
        full_text (str): Full text of the article.
    Returns:
        dict: Dictionary with article ID, first 500 words, and last 500 words.
    """
    cleaned_text = clean_text(full_text)
    first_500_words, last_500_words = split_text_words(cleaned_text)  # Split into word segments
    return {
        "article_id": article_id,
        "prompt": first_500_words,  # First 500 words as prompt for the model
        "ground_truth": last_500_words  # Last 500 words for comparison with model output
    }

# Main function to process the database
def process_database(input_file: str, output_file: str, sample_size: int = 100) -> None:
    """
    Read articles from a file, randomly select a sample, process each, and save to JSON.
    
    Args:
        input_file (str): Path to the input file containing all articles.
        output_file (str): Path to save the processed JSON output.
        sample_size (int): Number of articles to randomly select (default 100).
    """
    # Initialize a dictionary to store articles by ID
    articles = {}
    
    # Read the input file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
            article_blocks = re.split(r'@@(400\d{4})', content)
            for i in range(1, len(article_blocks), 2): 
                article_id = article_blocks[i]
                article_text = article_blocks[i + 1].strip()
                articles[article_id] = article_text
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Check if there are enough articles
    if len(articles) < sample_size:
        print(f"Warning: Only {len(articles)} articles available, less than {sample_size} requested.")
        sample_size = len(articles)
    
    # Randomly select 100 article IDs
    selected_ids = random.sample(list(articles.keys()), sample_size)
    
    # Process each selected article
    processed_articles = []
    for article_id in selected_ids:
        processed = process_article(article_id, articles[article_id])
        processed_articles.append(processed)
    
    # Save the processed data to a JSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_articles, f, indent=4, ensure_ascii=False)
        print(f"Successfully processed {len(processed_articles)} articles and saved to '{output_file}'")
    except Exception as e:
        print(f"Error writing to output file: {e}")


input_file = "input file's path"  
output_file = "output file path in json format"  

# Run the processing
process_database(input_file, output_file, sample_size=100)

# set token size
def preprocess_text(text):
    """Tokenize text into words and sentences."""
    try:
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        return sentences, words
    except Exception as e:
        print(f"Error in preprocessing text: {e}")
        return [], []

def truncate_to_n_tokens_complete_sentences(text, n=500):
    """Truncate text to complete sentences before exceeding n tokens and return text with token count."""
    try:
        sentences = sent_tokenize(text)
        token_count = 0
        truncated_sentences = []
        for sentence in sentences:
            words = word_tokenize(sentence)
            if token_count + len(words) <= n:
                truncated_sentences.append(sentence)
                token_count += len(words)
            else:
                break
        truncated_text = ' '.join(truncated_sentences)
        words = word_tokenize(truncated_text)
        num_tokens = len(words)
        if num_tokens == 0:
            print(f"Warning: No complete sentences found within {n} tokens.")
        elif num_tokens < n:
            print(f"Info: Complete sentences yield {num_tokens} tokens, less than {n}.")
        return truncated_text, num_tokens
    except Exception as e:
        print(f"Error in truncating text: {e}")
        return text, 0

def load_json_file(file_path):
    """Load and return data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return []

def store_truncated_texts(processed_path, generated_path, output_path):
    """Process texts from two JSON files and store truncated texts with token sizes."""
    try:
        # Load JSON files
        processed_data = load_json_file(processed_path)
        generated_data = load_json_file(generated_path)
        
        # Create a dictionary to map article_id to generated_continuation
        generated_dict = {item['article_id']: item['generated_continuation'] for item in generated_data}
        
        # Results storage
        results = []
        
        # Process each entry in processed_articles
        for item in processed_data:
            article_id = item.get('article_id')
            ground_truth = item.get('ground_truth', '')
            
            if not isinstance(ground_truth, str):
                print(f"Warning: ground_truth for article_id {article_id} is not a string, skipping.")
                continue
            
            # Get corresponding generated_continuation
            generated_text = generated_dict.get(article_id, '')
            if not isinstance(generated_text, str):
                print(f"Warning: generated_continuation for article_id {article_id} is not a string, skipping.")
                continue
            
            # Get truncated texts and token sizes
            ground_truncated, ground_token_size = truncate_to_n_tokens_complete_sentences(ground_truth, n=500)
            generated_truncated, generated_token_size = truncate_to_n_tokens_complete_sentences(generated_text, n=500)
            
            # Store results
            results.append({
                'article_id': article_id,
                'ground_truth_truncated_text': ground_truncated,
                'ground_truth_token_size': ground_token_size,
                'generated_continuation_truncated_text': generated_truncated,
                'generated_continuation_token_size': generated_token_size
            })
        
        # Save results to a JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        print(f"Truncated texts saved to {output_path}")
        
        return results
    except Exception as e:
        print(f"Error in processing texts: {e}")
        return []

# File paths
processed_path = r"add_a_local_path.json"
generated_path = r"add_a_local_path.json"
output_path = r"add_a_local_path.json"

# Ensure NLTK resources are available
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")

# Run the processing
if __name__ == "__main__":
    if os.path.exists(processed_path) and os.path.exists(generated_path):
        results = store_truncated_texts(processed_path, generated_path, output_path)
    else:
        print("Error: One or both input files do not exist.")