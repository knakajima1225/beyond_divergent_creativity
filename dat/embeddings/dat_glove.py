"""
This notebook is adapted from Bellemare-Pepin, DAT_GPT: https://github.com/AntoineBellemare/DAT_GPT
GloVe from https://nlp.stanford.edu/projects/glove/
"""
import re
import itertools
import numpy
import scipy.spatial.distance
import nltk
nltk.download('averaged_perceptron_tagger_eng')
from nltk.corpus import wordnet

class Model:
    """Create model to compute DAT"""

    def __init__(self, model, dictionary, pattern="^[a-z][a-z-]*[a-z]$"):
        """Join model and words matching pattern in dictionary
       
        Args:
        model : str or Path
            Path to the GloVe embedding file.
        dictionary : str or Path
            Path to the vocabulary file.
        pattern : str
            Regex pattern for valid words.
        """

        # Keep unique words matching pattern from file
        words = set()
        with open(dictionary, "r", encoding="utf8") as f:
            for line in f:
                if re.match(pattern, line):
                    words.add(line.rstrip("\n"))

        # Join words with model
        vectors = {}
        with open(model, "r", encoding="utf8") as f:
            for line in f:
                tokens = line.split(" ")
                word = tokens[0]
                if word in words:
                    vector = numpy.asarray(tokens[1:], "float32")
                    vectors[word] = vector
        self.vectors = vectors


    def validate(self, word):
        """Clean up word and find best candidate to use"""

        # Strip unwanted characters
        clean = re.sub(r"[^a-zA-Z- ]+", "", word).strip().lower()
        
        if len(clean) <= 1:
            return None # Word too short

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

        for cand in candidates:
            if cand in self.vectors:

                # Check if word is a noun
                word_tagged = nltk.pos_tag([cand])[0][1]

                # Added - For multiple possible tags, better to use WordNet
                possible_pos = set()
                for synset in wordnet.synsets(cand):
                    possible_pos.add(synset.pos())

                if word_tagged in ['NN','NNS'] or 'n' in possible_pos:
                    return cand
                else:
                    continue

        return None # Could not find valid word

    def distance(self, word1, word2):
        """Compute cosine distance (0 to 2) between two words"""
        vector1 = self.vectors.get(word1)
        vector2 = self.vectors.get(word2)
        
        # Check if both words exist in the model
        if vector1 is None: # not called
            print(f"Warning: '{word1}' not found in vectors")
            return None
        if vector2 is None: # not called
            print(f"Warning: '{word2}' not found in vectors")
            return None
        return scipy.spatial.distance.cosine(self.vectors.get(word1), self.vectors.get(word2))


    def dat(self, words, minimum=7):
        """Compute DAT score"""
        # Keep only valid unique words
        uniques = []
        for word in words:
            valid = self.validate(word)
            if valid and valid not in uniques:
                uniques.append(valid)

        # Keep subset of words
        n_uniques = len(uniques)
        if n_uniques < minimum:
            return (None, n_uniques)  # Not enough valid words

        subset = uniques[:minimum]

        # Compute distances between each pair of words 
        distances = [
            dist
            for w1, w2 in itertools.combinations(subset, 2)
            if (dist := self.distance(w1, w2)) is not None
        ]

        # Check if we have enough valid distances
        if not distances:
            return (None, n_uniques)

        self.distances = distances

        # Compute the DAT score
        score = (sum(distances) / len(distances)) * 100
        return (score, n_uniques)