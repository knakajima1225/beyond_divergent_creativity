"""
Script to process cue words and generate common word associations using GPT API.
"""

import argparse
import json
import os
from openai import OpenAI
import sys
from pathlib import Path
from cdat.embeddings import cdat_sbert

BATCH_SIZE = 50          # Process 50 samples at a time
TOTAL_ITERATIONS = 550    # Cut-off
cut_off = TOTAL_ITERATIONS
results_file = f"common_results{cut_off}_test.json"

client = OpenAI(
    api_key=""  # or rely on environment variable
)


def sample_words(cue):
    """Generate 10 common noun associations for a cue word"""
    prompt = (
        f'Please enter 10 words that are most semantically associated with the '
        f'following cue word: "{cue}." '
        'Rules: Only single words in English. '
        'Only nouns (e.g., things, objects, concepts). '
        'No proper nouns (e.g., no specific people or places). '
        'No specialized vocabulary (e.g., no technical terms). '
        'Think of the words on your own. '
        'Make a list of these 10 words, one per line.'
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        text = response.choices[0].message.content.strip()
        # Remove numbering if present
        words = [line.lstrip("0123456789. ").strip() for line in text.splitlines()]
        return words

    except Exception as e:
        print(f"Error processing cue '{cue}': {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Generate common word associations using GPT and compute CDAT scores."
    )
    parser.add_argument(
        "cue_file",
        type=str,
        help="Path to the cue file (one cue word per line)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.cue_file):
        raise FileNotFoundError(f"Cue file not found: {args.cue_file}")

    # Read cues
    with open(args.cue_file, "r") as f:
        cues = f.read().splitlines()

    print(f"Loaded {len(cues)} cues")

    # Initialize SBERT model
    print("Initializing SBERT model...")
    sbert_model = cdat_sbert.Model()

    common_results = []

    for batch_start in range(0, len(cues), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(cues))
        print(
            f"Processing batch {batch_start // BATCH_SIZE + 1}: "
            f"samples {batch_start} to {batch_end - 1}"
        )

        for i in range(batch_start, batch_end):
            cue = cues[i]

            common_words = sample_words(cue)
            print(f"cue: {cue}, sample: {common_words}")

            score, num_valid = sbert_model.cdat(cue, common_words)
            divergence, appropriateness = score

            common_results.append({
                "Cue": cue,
                "Words": common_words,
                "sbert_novelty": divergence,
                "sbert_appropriateness": appropriateness,
                "sbert_valid_count": num_valid,
                "Model": "Common",
            })

        # Save intermediate results after each batch
        with open(results_file, "w") as f:
            json.dump(common_results, f, indent=2)

        print(f"Saved {len(common_results)} results to {results_file}")

    print("Processing complete.")


if __name__ == "__main__":
    main()
