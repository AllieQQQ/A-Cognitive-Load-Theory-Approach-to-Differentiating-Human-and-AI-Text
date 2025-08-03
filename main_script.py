import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
from math import log, sqrt
from collections import Counter
import random
import json
import pandas as pd
import os
import json
import spacy
from collections import defaultdict
import re

# calculate basic metrics
def preprocess_text(text):
    """Tokenize text into words and sentences."""
    try:
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        return sentences, words
    except Exception as e:
        print(f"Error in preprocessing text: {e}")
        return [], []

def truncate_to_n_tokens_complete_sentences(text, n=400):
    """Truncate text to complete sentences before exceeding n tokens."""
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
        if len(words) == 0:
            print(f"Warning: No complete sentences found within {n} tokens.")
        elif len(words) < n:
            print(f"Info: Complete sentences yield {len(words)} tokens, less than {n}.")
        return truncated_text
    except Exception as e:
        print(f"Error in truncating text: {e}")
        return text

def compute_length_metrics(text):
    """Compute four linguistic metrics from the text."""
    try:
        truncated_text = truncate_to_n_tokens_complete_sentences(text, n=400)
        sentences, words = preprocess_text(truncated_text)
        
        # Count letters and alphabetic words for letters_per_word
        num_letters = sum(len(word) for word in words if word.isalpha())
        num_alpha_words = sum(1 for word in words if word.isalpha())
        
        # Total number of tokens (all tokens)
        num_tokens = len(words)
        
        # Total number of sentences
        num_sentences = len(sentences) if sentences else 0
        
        return {
            "letters_per_word": round(num_letters / num_alpha_words, 4) if num_alpha_words else 0,
            "words_per_sentence": round(num_tokens / num_sentences, 4) if num_sentences else 0,
            "words_per_text": num_tokens,
            "sentences_per_text": num_sentences
        }
    except Exception as e:
        print(f"Error in computing metrics: {e}")
        return {
            "letters_per_word": 0,
            "words_per_sentence": 0,
            "words_per_text": 0,
            "sentences_per_text": 0
        }

def load_json_file(file_path):
    """Load and return data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return []

def process_texts(processed_path, generated_path, output_path):
    """Process texts from two JSON files, compute metrics, and save results."""
    try:
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
            
            # Compute metrics for both texts
            ground_metrics = compute_length_metrics(ground_truth)
            generated_metrics = compute_length_metrics(generated_text)
            
            # Store results for comparison
            results.append({
                'article_id': article_id,
                'ground_truth_metrics': ground_metrics,
                'generated_continuation_metrics': generated_metrics
            })
        
        # Save results to a JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_path}")
        
        return results
    except Exception as e:
        print(f"Error in processing texts: {e}")
        return []

# File paths
processed_path = r"add_a_local_path"
generated_path = r"add_a_local_path"
output_path = r"add_a_local_path"

try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")

# Run the processing
if __name__ == "__main__":
    if os.path.exists(processed_path) and os.path.exists(generated_path):
        results = process_texts(processed_path, generated_path, output_path)
    else:
        print("Error: One or both input files do not exist.")

# calculate lexical metrics
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Load top 2,000 frequent words and POS tags
try:
    freq_pos_dict = {}
    with open(r'add the frequent words file', 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:  # Expect word and POS
                word, pos = parts[0].lower(), parts[1]
                freq_pos_dict[word] = pos
            else:
                print(f"Warning: Skipping malformed line: {line.strip()}")
except FileNotFoundError:
    print("Error: 'frequency and pos.txt' not found")
    freq_pos_dict = {}

# Map custom POS tags to NLTK Penn Treebank tags
pos_map = {
    'a': 'DT',  
    'v': 'VB',  
    'c': 'CC',  
    'i': 'IN',  
    't': 'TO',  
    'p': 'PRP', 
    'd': 'DT',  
    'x': 'RB'   
}

def is_content_word(tag):
    return tag.startswith(('NN', 'VB', 'JJ', 'RB'))

def get_word_types(tokens):
    return set(tokens)

def is_sophisticated(word, freq_dict):
    return word not in freq_dict

def get_custom_pos(word, pos_dict, pos_map):
    if word in pos_dict:
        custom_pos = pos_dict[word]
        return pos_map.get(custom_pos, 'NN')
    return None

def calculate_metrics(text):
    # Tokenize and POS tag
    tokens = word_tokenize(text.lower())
    tagged = pos_tag(tokens)

    # Override NLTK POS tags with custom POS
    custom_tagged = []
    for token, nltk_tag in tagged:
        custom_pos = get_custom_pos(token, freq_pos_dict, pos_map)
        custom_tagged.append((token, custom_pos if custom_pos else nltk_tag))

    content_tokens = [t for t, tag in custom_tagged if is_content_word(tag)]
    ld = len(content_tokens) / len(tokens) if tokens else 0

    sophisticated_tokens = [t for t in tokens if is_sophisticated(t, freq_pos_dict)]
    ls1 = len(sophisticated_tokens) / len(tokens) if tokens else 0

    types = get_word_types(tokens)
    sophisticated_types = [t for t in types if is_sophisticated(t, freq_pos_dict)]
    ls2 = len(sophisticated_types) / len(types) if types else 0

    verbs = [t for t, tag in custom_tagged if tag.startswith('VB')]
    verb_types = get_word_types(verbs)
    sophisticated_verbs = [v for v in verbs if is_sophisticated(v, freq_pos_dict)]
    sophisticated_verb_types = [v for v in verb_types if is_sophisticated(v, freq_pos_dict)]

    vs1 = len(sophisticated_verbs) / len(verbs) if verbs else 0
    vs2 = len(sophisticated_verb_types) / len(verb_types) if verb_types else 0
    cvs1 = (len(sophisticated_verb_types) ** 2) / len(verbs) if verbs else 0

    t = len(types)
    t50 = len(get_word_types(tokens[:50]))

    t50s_samples = []
    for _ in range(10):
        if len(tokens) >= 50:
            sample = random.sample(tokens, 50)
            t50s_samples.append(len(get_word_types(sample)))
        else:
            t50s_samples.append(len(types))
    t50s = sum(t50s_samples) / len(t50s_samples) if t50s_samples else 0

    t50w_samples = []
    for i in range(10):
        start = i * 50
        if start < len(tokens):
            sample = tokens[start:start + 50]
            t50w_samples.append(len(get_word_types(sample)))
        else:
            t50w_samples.append(len(types))
    t50w = sum(t50w_samples) / len(t50w_samples) if t50w_samples else 0

    ttr = len(types) / len(tokens) if tokens else 0

    segments = [tokens[i:i + 50] for i in range(0, len(tokens), 50)]
    msttr_values = [len(get_word_types(seg)) / len(seg) for seg in segments if seg]
    msttr = sum(msttr_values) / len(msttr_values) if msttr_values else 0

    cttr = len(types) / sqrt(2 * len(tokens)) if tokens else 0
    rttr = len(types) / sqrt(len(tokens)) if tokens else 0
    logttr = log(len(types)) / log(len(tokens)) if tokens and len(tokens) > 1 else 0
    uber = log(len(types)) / log(log(len(tokens))) if tokens and len(tokens) > 1 and log(len(tokens)) > 0 else 0

    window_size = 10
    mattr_values = []
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i + window_size]
        mattr_values.append(len(get_word_types(window)) / len(window) if window else 0)
    mattr = sum(mattr_values) / len(mattr_values) if mattr_values else 0

    hdd = ttr 

    def calc_mtld(token_list, threshold=0.72):
        if not token_list:
            return 0
        factor_count = 0
        current_tokens = []
        for t in token_list:
            current_tokens.append(t)
            current_ttr = len(get_word_types(current_tokens)) / len(current_tokens)
            if current_ttr <= threshold or len(current_tokens) == len(token_list):
                factor_count += 1
                current_tokens = []
        return len(token_list) / factor_count if factor_count else 0
    mtld = calc_mtld(tokens)
    mtld_ma = mtld
    mtld_backward = calc_mtld(tokens[::-1])
    mtld_bii = (mtld + mtld_backward) / 2 if tokens else 0

    lexical_tokens = content_tokens
    lexical_types = get_word_types(lexical_tokens)
    lv = len(lexical_types) / len(lexical_tokens) if lexical_tokens else 0

    vv1 = len(verb_types) / len(verbs) if verbs else 0
    svv1 = (len(verb_types) ** 2) / len(verbs) if verbs else 0
    cvv1 = vv1
    cvv2 = vv1

    nouns = [t for t, tag in custom_tagged if tag.startswith('NN')]
    noun_types = get_word_types(nouns)
    nv = len(noun_types) / len(nouns) if nouns else 0

    adjectives = [t for t, tag in custom_tagged if tag.startswith('JJ')]
    adj_types = get_word_types(adjectives)
    adjv = len(adj_types) / len(adjectives) if adjectives else 0

    adverbs = [t for t, tag in custom_tagged if tag.startswith('RB')]
    adv_types = get_word_types(adverbs)
    advv = len(adv_types) / len(adverbs) if adverbs else 0

    modifiers = adjectives + adverbs
    mod_types = get_word_types(modifiers)
    modv = len(mod_types) / len(modifiers) if modifiers else 0

    return {
        'LD': ld, 'LS1': ls1, 'LS2': ls2, 'VS1': vs1, 'VS2': vs2, 'CVS1': cvs1,
        'T': t, 'T50': t50, 'T50S': t50s, 'T50W': t50w, 'TTR': ttr, 'MSTTR': msttr,
        'CTTR': cttr, 'RTTR': rttr, 'LogTTR': logttr, 'Uber': uber, 'MATTR': mattr,
        'HDD': hdd, 'MTLD': mtld, 'MTLD-MA': mtld_ma, 'MTLD-bii': mtld_bii, 'LV': lv,
        'VV1': vv1, 'SVV1': svv1, 'CVV1': cvv1, 'CVV2': cvv2, 'NV': nv, 'ADJV': adjv,
        'ADVV': advv, 'MODV': modv
    }

# Load JSON file
json_path = r'add the local file'
try:
    with open(json_path, 'r') as file:
        data = json.load(file)
except FileNotFoundError:
    print(f"Error: {json_path} not found")
    exit(1)
except json.JSONDecodeError:
    print(f"Error: Invalid JSON format in {json_path}")
    exit(1)

# Process texts and compute metrics
results = []
for entry in data:
    article_id = entry.get('article_id')
    ground_text = entry.get('ground_truth_truncated_text', '')
    generated_text = entry.get('generated_continuation_truncated_text', '')

    if not article_id or not ground_text or not generated_text:
        print(f"Warning: Skipping entry with missing article_id or text: {article_id}")
        continue

    # Ground truth metrics
    ground_metrics = calculate_metrics(ground_text)
    ground_row = {'article_id': article_id, 'text_type': 'ground_truth'}
    ground_row.update(ground_metrics)
    results.append(ground_row)

    # Generated metrics
    generated_metrics = calculate_metrics(generated_text)
    generated_row = {'article_id': article_id, 'text_type': 'generated'}
    generated_row.update(generated_metrics)
    results.append(generated_row)

    # Difference (ground - generated)
    diff_row = {'article_id': article_id, 'text_type': 'difference'}
    for metric in ground_metrics:
        diff_row[metric] = ground_metrics[metric] - generated_metrics[metric]
    results.append(diff_row)

# Save to CSV
output_path = r'a_local_path'
df = pd.DataFrame(results)
df = df.round(3)  # Round to 3 decimal places
try:
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
except Exception as e:
    print(f"Error saving CSV: {e}")

# Display sample output
print("\nSample Results (first few rows):")
print(df.head(9))  # Show first 3 articles (ground, generated, difference)

# calculate syntactic metrics
# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

def count_words(text):
    """Count total words (tokens excluding punctuation)."""
    doc = nlp(text)
    return sum(1 for token in doc if not token.is_punct)

def count_sentences(text):
    """Count sentences using spaCy."""
    doc = nlp(text)
    return len(list(doc.sents))

def identify_t_units(doc):
    """Identify T-units (main clause + dependent clauses)."""
    t_units = []
    for sent in doc.sents:
        main_clauses = []
        current_t_unit = []
        for token in sent:
            if token.dep_ in ("ROOT", "conj") and token.pos_ == "VERB":
                if current_t_unit:
                    main_clauses.append(current_t_unit)
                    current_t_unit = []
                current_t_unit.append(token)
            else:
                current_t_unit.append(token)
        if current_t_unit:
            main_clauses.append(current_t_unit)
        t_units.extend(main_clauses)
    return t_units

def count_clauses(doc):
    """Count clauses (main and dependent)."""
    clauses = 0
    for token in doc:
        if token.dep_ in ("ROOT", "ccomp", "advcl", "relcl", "acl"):
            clauses += 1
    return clauses

def count_dependent_clauses(doc):
    """Count dependent clauses."""
    dependent_clauses = 0
    for token in doc:
        if token.dep_ in ("ccomp", "advcl", "relcl", "acl"):
            dependent_clauses += 1
    return dependent_clauses

def count_verb_phrases(doc):
    """Count verb phrases (approximated by verbs with their subtrees)."""
    verb_phrases = 0
    for token in doc:
        if token.pos_ == "VERB" and token.dep_ in ("ROOT", "ccomp", "advcl", "relcl", "acl", "conj"):
            verb_phrases += 1
    return verb_phrases

def count_coordinate_phrases(doc):
    """Count coordinate phrases (phrases joined by coordinating conjunctions)."""
    coordinate_phrases = 0
    for token in doc:
        if token.dep_ == "cc" and token.text.lower() in ("and", "or", "but"):
            if token.head.pos_ in ("NOUN", "VERB", "ADJ", "ADV"):
                coordinate_phrases += 1
    return coordinate_phrases

def count_complex_nominals(doc):
    """Count complex nominals (noun phrases with modifiers or clauses)."""
    complex_nominals = 0
    for chunk in doc.noun_chunks:
        has_dependent_clause = any(token.dep_ in ("relcl", "acl") for token in chunk)
        has_multiple_modifiers = len([t for t in chunk if t.dep_ in ("amod", "compound")]) > 1
        if has_dependent_clause or has_multiple_modifiers:
            complex_nominals += 1
    return complex_nominals

def calculate_syntactic_metrics(text):
    """Calculate the 14 syntactic complexity metrics."""
    doc = nlp(text)

    # Basic counts
    total_words = count_words(text)
    total_sentences = count_sentences(text)
    t_units = identify_t_units(doc)
    total_t_units = len(t_units)
    total_clauses = count_clauses(doc)
    total_dependent_clauses = count_dependent_clauses(doc)
    total_verb_phrases = count_verb_phrases(doc)
    total_coordinate_phrases = count_coordinate_phrases(doc)
    total_complex_nominals = count_complex_nominals(doc)

    # Calculate metrics
    metrics = {}

    # Units of Production
    metrics["MLS"] = total_words / total_sentences if total_sentences > 0 else 0
    metrics["MLT"] = total_words / total_t_units if total_t_units > 0 else 0
    metrics["MLC"] = total_words / total_clauses if total_clauses > 0 else 0

    # Complexity Ratios
    metrics["C/S"] = total_clauses / total_sentences if total_sentences > 0 else 0
    metrics["VP/T"] = total_verb_phrases / total_t_units if total_t_units > 0 else 0
    metrics["C/T"] = total_clauses / total_t_units if total_t_units > 0 else 0
    metrics["DC/C"] = total_dependent_clauses / total_clauses if total_clauses > 0 else 0
    metrics["DC/T"] = total_dependent_clauses / total_t_units if total_t_units > 0 else 0
    metrics["CT/T"] = sum(1 for t_unit in t_units if any(token.dep_ in ("ccomp", "advcl", "relcl", "acl") for token in t_unit)) / total_t_units if total_t_units > 0 else 0
    metrics["CP/T"] = total_coordinate_phrases / total_t_units if total_t_units > 0 else 0
    metrics["CP/C"] = total_coordinate_phrases / total_clauses if total_clauses > 0 else 0
    metrics["CN/T"] = total_complex_nominals / total_t_units if total_t_units > 0 else 0
    metrics["CN/C"] = total_complex_nominals / total_clauses if total_clauses > 0 else 0
    metrics["T/S"] = total_t_units / total_sentences if total_sentences > 0 else 0

    return metrics

sample_text = """
a sample text
"""
metrics = calculate_syntactic_metrics(sample_text)
for metric, value in metrics.items():
    print(f"{metric}: {value:.2f}")

# use the code to process all data and get the syntatic metrics results
