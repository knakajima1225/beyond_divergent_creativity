"""
Process baseline files and compute CDAT scores based on embeddings using GloVe, FastText, and BERT layers (3-9 + avg).
Usage: python baseline_embeddings.py <input_json_file>
Example: python baseline_embeddings.py random_results550.json
"""

import sys
import os
import json
import pandas as pd
import ast
from tqdm import tqdm

# Import the model classes
from cdat.embeddings import (
    cdat_fasttext,
    cdat_glove,
    cdat_sbert,
    cdat_bert_layers,
)

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2] 
FASTTEXT_PATH = REPO_ROOT / "model" / "fasttext" / "cc.en.300.bin"
GLOVE_PATH = REPO_ROOT / "model" / "glove" / "glove.840B.300d.txt"

# ============================================================================
# Parse command-line argument
# ============================================================================
if len(sys.argv) < 2:
    print("Usage: python -m cdat.scripts.baseline_embeddings.py <input_json_file>")
    print("Example: python -m cdat.scripts.baseline_embeddings.py cdat/data/baseline/random_results550.json")
    sys.exit(1)

input_file = sys.argv[1]

# Check if input file exists
if not os.path.exists(input_file):
    print(f"Error: Input file '{input_file}' not found")
    sys.exit(1)

# Derive output filename from input filename
input_basename = os.path.splitext(os.path.basename(input_file))[0]
output_file = f'results_with_embeddings_{input_basename}_baselines.csv'

# ============================================================================
# RESET: Clear variables and files at the start
# ============================================================================
# Reset dataframes
combined_df = None
new_df = None
new_rows = []

# Delete output file if it exists
if os.path.exists(output_file):
    os.remove(output_file)
    print(f"   ✓ Deleted existing output file: {output_file}")

# ============================================================================
# LOAD DATA
# ============================================================================
# Load JSON file
print(f"Loading JSON file: {input_file}")
with open(input_file, 'r') as f:
    results = json.load(f)
combined_df = pd.DataFrame(results)

# Initialize models
bert_model = cdat_bert_layers.Model(model_name='bert-base-uncased')
fasttext_model = cdat_fasttext.Model(model_name=str(FASTTEXT_PATH))
glove_model = cdat_glove.Model(model_path=str(GLOVE_PATH))

# Get layers from bert_model (layers 3-9 + 'avg')
layers = bert_model.layers + ['avg']

# Prepare new dataframe with required columns
new_rows = []

# Process each row
for idx, row in tqdm(combined_df.iterrows(), total=len(combined_df), desc="Processing rows"):
    # Extract required fields (adjust column names based on actual JSON structure)
    temperature = None
    model = row.get('Model', None)
    words = row.get('Words', [])  # Assuming Words is a list
    cue = row.get('Cue', None)
    sbert_novelty = row.get('sbert_novelty', None)
    sbert_appropriateness = row.get('sbert_appropriateness', None)
    sbert_valid_count = row.get('sbert_valid_count', None)
    
    # Handle Words if it's a string that needs parsing
    if isinstance(words, str):
        try:
            # Try parsing as JSON first
            words = json.loads(words)
        except:
            try:
                # Try parsing as Python literal (list)
                words = ast.literal_eval(words)
            except:
                # Fall back to comma-separated string
                words = [w.strip() for w in words.split(',')] if words else []
    
    # Ensure words is a list (convert to empty list if not)
    if not isinstance(words, list):
        words = []
    
    # Ensure cue is a string (convert None to None, other types to string)
    if cue is not None and not isinstance(cue, str):
        cue = str(cue)
    
    # Count total words
    count = len(words) if isinstance(words, list) else 0
    
    # Calculate metrics for each model
    # Initialize BERT layer results
    bert_layer_results = {f'bert_l{layer}_novelty': None for layer in layers}
    bert_layer_results.update({f'bert_l{layer}_appropriateness': None for layer in layers})
    bert_valid_count = None
    
    fasttext_novelty = None
    fasttext_appropriateness = None
    fasttext_valid_count = None
    glove_novelty = None
    glove_appropriateness = None
    glove_valid_count = None
    
    if cue and words:
        # BERT Layers
        try:
            bert_results_dict, bert_valid_count = bert_model.cdat(cue, words)
            # bert_results_dict is a dictionary with keys for each layer (3-9, 'avg')
            # Each value is a tuple (novelty, appropriateness)
            for layer in layers:
                if layer in bert_results_dict and bert_results_dict[layer][0] is not None:
                    bert_layer_results[f'bert_l{layer}_novelty'] = bert_results_dict[layer][0]
                    bert_layer_results[f'bert_l{layer}_appropriateness'] = bert_results_dict[layer][1]
        except Exception as e:
            print(f"Error calculating BERT layers for row {idx}: {e}")
            bert_valid_count = None
        
        # FastText
        try:
            fasttext_result, fasttext_valid_count = fasttext_model.cdat(cue, words)
            if fasttext_result[0] is not None:
                fasttext_novelty = fasttext_result[0]  # Divergence (novelty)
                fasttext_appropriateness = fasttext_result[1]  # 200 - Dissimilarity
        except Exception as e:
            print(f"Error calculating FastText for row {idx}: {e}")
            fasttext_valid_count = None
        
        # GloVe
        try:
            glove_result, glove_valid_count = glove_model.cdat(cue, words)
            if glove_result[0] is not None:
                glove_novelty = glove_result[0]  # Divergence (novelty)
                glove_appropriateness = glove_result[1]  # 200 - Dissimilarity
        except Exception as e:
            print(f"Error calculating GloVe for row {idx}: {e}")
            glove_valid_count = None
    
    # Create new row
    # Convert Words list to JSON string for CSV storage
    words_str = json.dumps(words) if isinstance(words, list) else json.dumps([])
    
    new_row = {
        'Temperature': temperature,
        'Model': model,
        'Words': words_str,
        'Cue': cue,
        'Count': count,
        'sbert_novelty': sbert_novelty,
        'sbert_appropriateness': sbert_appropriateness,
        'sbert_valid_count': sbert_valid_count,
        'fasttext_novelty': fasttext_novelty,
        'fasttext_appropriateness': fasttext_appropriateness,
        'fasttext_valid_count': fasttext_valid_count,
        'glove_novelty': glove_novelty,
        'glove_appropriateness': glove_appropriateness,
        'glove_valid_count': glove_valid_count,
        'bert_valid_count': bert_valid_count
    }
    
    # Add BERT layer results to the row
    new_row.update(bert_layer_results)
    
    new_rows.append(new_row)
    
    # Clear cache periodically to save memory
    if (idx + 1) % 100 == 0:
        bert_model.clear_cache()
        fasttext_model.clear_cache()

# Create new dataframe
new_df = pd.DataFrame(new_rows)

# Save to file
new_df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")
print(f"Shape: {new_df.shape}")
print(f"Sample results:")
print(new_df.head())
