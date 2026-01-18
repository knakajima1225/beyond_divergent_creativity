"""
Generate random baseline results for CDAT evaluation.
This script samples random words from WordNet and computes CDAT scores using SBERT embeddings.
"""

import argparse
import json
import os
import random
import nltk
from nltk.corpus import wordnet as wn
import cdat_bert_sbert

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate random baseline results for CDAT evaluation')
parser.add_argument('cue_file', type=str, help='Path to the cue file (one cue per line)')
args = parser.parse_args()

# Be sure to set random seed for reproducibility
# NOTE: the original code was run without a seed. The dataset is available in the repository. 
#random.seed(42)

# Download WordNet if not already installed
nltk.download('wordnet', quiet=True)

# Set batch size and total iterations
BATCH_SIZE = 50  # Process 50 samples at a time
TOTAL_ITERATIONS = 50
cut_off = TOTAL_ITERATIONS

# Collect valid words from WordNet
# Only nouns, excluding multi-word phrases, proper nouns, and numbers
valid_words = [
    lemma.name()  # Extract the word
    for synset in wn.all_synsets(pos=wn.NOUN)  # Only nouns
    for lemma in synset.lemmas()  # Get all lemma names
    if "_" not in lemma.name()  # Exclude multi-word phrases
    and not lemma.name()[0].isupper()  # Exclude capitalized words (proper nouns)
    and not lemma.name().isdigit()  # Exclude words that are only numbers
]

# Initialize results list
results_file = f'random_results{cut_off}_test.json'
random_results = []

# Load cues from file (provided as command line argument)
with open(args.cue_file, 'r') as f:
    cues = f.read().splitlines()  # splitlines() removes the newline characters
cues = cues[:TOTAL_ITERATIONS]

# Initialize SBERT model for computing CDAT scores
sbert_model = cdat_sbert.Model(model_name='all-MiniLM-L6-v2')

# Process in batches
for batch_start in range(0, cut_off, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, cut_off)
    print(f"Processing batch {batch_start//BATCH_SIZE + 1}: samples {batch_start} to {batch_end-1}")

    batch_results = []
    for i in range(batch_start, batch_end):
        cue = cues[i]

        # Sample 10 random words
        random_words = random.sample(valid_words, 10)
        print(f"cue: {cue}, sample: {random_words}")

        # Compute CDAT scores using SBERT model
        # Returns: ((novelty, appropriateness), num_valid)
        score_tuple, num_valid = sbert_model.cdat(cue, random_words)
        
        # Extract novelty and appropriateness scores
        if score_tuple[0] is not None and score_tuple[1] is not None:
            novelty, appropriateness = score_tuple
        else:
            novelty = None
            appropriateness = None

        # Store results with renamed columns
        batch_results.append({
            'Cue': cue,
            'Words': random_words,
            'sbert_novelty': novelty,  # Renamed from Divergence
            'sbert_appropriateness': appropriateness,  # Computed as 200-Dissimilarity in the model
            'sebert_valid_count': num_valid,  # Renamed from Valid count
            'Model': 'Random',
        })
    
    # Append batch results to existing results
    random_results.extend(batch_results)
    
    # Save results after each batch
    with open(results_file, 'w') as f:
        json.dump(random_results, f, indent=2)
    
    print(f"Saved results up to sample {batch_end-1}")
    
    # Clear memory
    del batch_results

print(f"Processing complete. Results saved to {results_file}")
