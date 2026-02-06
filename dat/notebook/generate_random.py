"""
Sample random words and compute DAT score using GloVe and SBERT.
"""
import os
import json
import random
import numpy as np

import nltk
from nltk.corpus import wordnet as wn

from dat.embeddings import dat_glove, dat_sbert
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# NOTE: Be sure to set random seed for reproducibility
# the original code was run without a seed. The dataset is available in the repository. 
#random.seed(42)

# -------------------------
# Configuration
# -------------------------
BATCH_SIZE = 50
TOTAL_ITERATIONS = 500
CUTOFF = TOTAL_ITERATIONS
RESULTS_FILE = REPO_ROOT / "dat" / "data" / f"random_results{CUTOFF}_test.json"


# -------------------------
# Setup
# -------------------------
def setup_nltk():
    """Download required NLTK resources if missing."""
    nltk.download("wordnet")
    nltk.download("omw-1.4")


def load_valid_words():
    """Collect valid WordNet noun lemmas."""
    return [
        lemma.name()
        for synset in wn.all_synsets(pos=wn.NOUN)
        for lemma in synset.lemmas()
        if "_" not in lemma.name()
        and not lemma.name()[0].isupper()
        and not lemma.name().isdigit()
    ]


# -------------------------
# Main logic
# -------------------------

def main():
    setup_nltk()

    print("Loading models...")
    sbert_model = dat_sbert.Model()
    glove_model = dat_glove.Model(
        model=REPO_ROOT / "model" / "glove" / "glove.840B.300d.txt",
        dictionary=REPO_ROOT / "model" / "glove" / "words_glove.txt",
    )

    print("Collecting valid words from WordNet...")
    valid_words = load_valid_words()
    print(f"Total valid words: {len(valid_words)}")

    # Load or initialize results
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            random_results = json.load(f)
        print(f"Loaded {len(random_results)} existing results.")
    else:
        random_results = []

    # Batch processing
    for batch_start in range(len(random_results), CUTOFF, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, CUTOFF)
        print(
            f"\nProcessing batch {batch_start // BATCH_SIZE + 1}: "
            f"samples {batch_start} to {batch_end - 1}"
        )

        batch_results = []

        for i in range(batch_start, batch_end):
            random_words = random.sample(valid_words, 10)
            print(f"Sample {i}: {random_words}")

            sbert_score, _ = sbert_model.dat(random_words)
            glove_score, _ = glove_model.dat(random_words)

            print(f"SBERT DAT score: {sbert_score}")

            # Convert NumPy floats to Python floats
            sbert_score = float(sbert_score) if sbert_score is not None else None
            glove_score = float(glove_score) if glove_score is not None else None

            batch_results.append({
                "words": random_words,
                "sbert_score": sbert_score,
                "glove_score": glove_score,
            })

        random_results.extend(batch_results)

        # Save checkpoint
        with open(RESULTS_FILE, "w") as f:
            json.dump(random_results, f, indent=2)

        print(f"Saved results up to sample {batch_end - 1}")

        # Explicit cleanup (useful for long runs)
        del batch_results


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    main()
