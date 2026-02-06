import re
import itertools
import numpy as np
import scipy.spatial.distance
import nltk
nltk.download('averaged_perceptron_tagger_eng')
from nltk.corpus import wordnet


class Model:
    """Create model to compute CDAT using GloVe"""

    def __init__(self, model_path, pattern="^[a-z][a-z-]*[a-z]$"):
        """Load GloVe model - Load all words matching pattern directly from GloVe file
        
        Args:
            model_path (str): Path to GloVe model file (.txt). 
            pattern (str): Regex pattern to match valid words.
                         Default: '^[a-z][a-z-]*[a-z]$' (lowercase words with optional hyphens)
        """
        # Load all words from GloVe model that match the pattern directly
        # Single pass through the file, no dictionary filtering needed
        vectors = {}
        with open(model_path, "r", encoding="utf8") as f:
            for line in f:
                tokens = line.split(" ")
                word = tokens[0]

                if re.match(pattern, word):
                    vector = np.asarray(tokens[1:], dtype="float32")
                    vectors[word] = vector
        self.vectors = vectors
        print(f"Loaded {len(vectors)} word vectors from GloVe model (matching pattern)")

    def validate(self, word):
        """Clean up word and ensure it's a valid noun
        
        Unlike FastText, GloVe cannot handle OOV (out-of-vocabulary) words,
        so we must check if the word exists in the vocabulary.
        """
        # Strip unwanted characters
        clean = re.sub(r"[^a-zA-Z- ]+", "", word).strip().lower()
        
        if len(clean) <= 1:
            return None  # Word too short

        # Generate candidates for possible compound words
        # "valid" -> ["valid"]
        # "cul de sac" -> ["cul-de-sac", "culdesac"]
        # "top-hat" -> ["top-hat", "tophat"]
        candidates = []
        if " " in clean:
            candidates.append(re.sub(r" +", "-", clean))
            candidates.append(re.sub(r" +", "", clean))
        else:
            candidates.append(clean)
            if "-" in clean:
                candidates.append(re.sub(r"-+", "", clean))

        # Try each candidate - must exist in vocabulary AND be a valid noun
        for cand in candidates:
            if cand in self.vectors:
                # Check if word is a noun using POS tagging
                word_tagged = nltk.pos_tag([cand])[0][1]
                
                # Added - For multiple possible tags, better to use WordNet
                possible_pos = set()
                for synset in wordnet.synsets(cand):
                    possible_pos.add(synset.pos())

                if word_tagged in ['NN','NNS'] or 'n' in possible_pos:
                    return cand
                else:
                    continue

        # No valid noun candidate found in vocabulary
        return None

    def embed(self, word):
        """Get word embedding using GloVe"""
        # GloVe: get word vector directly from dictionary (word must exist in vocabulary)
        if word not in self.vectors:
            raise KeyError(f"Word '{word}' not found in GloVe vocabulary. Use validate() first.")
        return self.vectors[word]
    
    def distance(self, word1, word2):
        """Compute cosine distance (0 to 2) between two words"""
        vec1, vec2 = self.embed(word1), self.embed(word2)
        return scipy.spatial.distance.cosine(vec1, vec2)

    def cdat(self, cue, words, minimum=7):
        """Compute CDAT score"""
        uniques = []
        for word in words:
            valid = self.validate(word)
            if valid and valid not in uniques:
                uniques.append(valid)

        num_unique = len(uniques)
        if num_unique < minimum:
            return ((None, None), num_unique)
        else:
            subset = uniques[:minimum]

            # average value of the semantic distance between the cue and each word
            # multiplied by 100, which ranges from 0 to 200 
            # (the more distant the higher the score). 
            novelty_distances = [self.distance(w1, w2) for w1, w2 in itertools.combinations(subset, 2)]
            novelty = float((sum(novelty_distances) / len(novelty_distances)) * 100)

            dissimilarity_distances = [self.distance(cue, w1) for w1 in subset]
            dissimilarity = float(sum(dissimilarity_distances) / len(dissimilarity_distances) * 100)
            appropriateness = 200 - dissimilarity

        return ((novelty, appropriateness), num_unique)

