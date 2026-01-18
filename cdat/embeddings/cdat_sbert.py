import re
import itertools
import numpy as np
import scipy.spatial.distance
from sentence_transformers import SentenceTransformer
import nltk
nltk.download('averaged_perceptron_tagger_eng')
from nltk.corpus import wordnet


class Model:
    """Create model to compute CDAT using SBERT (Sentence-BERT)"""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Load SBERT-based model
        
        Args:
            model_name (str): SBERT model name/identifier
                - Examples: 'all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'paraphrase-MiniLM-L6-v2'
                - See: https://www.sbert.net/docs/pretrained_models.html
        """
        self.model_name = model_name
        self.vectors = {}
        self.model = SentenceTransformer(model_name)

    def validate(self, word):
        """Clean up word and ensure it's a valid noun"""
        # Strip unwanted characters
        clean = re.sub(r"[^a-zA-Z- ]+", "", word).strip().lower()
        
        if len(clean) <= 1:
            return None  # Word too short

        # Check if word is a noun using POS tagging
        word_tagged = nltk.pos_tag([clean])[0][1]
        
        # For multiple possible tags, better to use WordNet
        possible_pos = set()
        for synset in wordnet.synsets(clean):
            possible_pos.add(synset.pos())

        if word_tagged in ['NN','NNS'] or 'n' in possible_pos:
            return clean
        else:
            return None

    def embed(self, word):
        """Get word embedding using SBERT"""
        if word not in self.vectors:
            self.vectors[word] = self.model.encode(word, convert_to_numpy=True)
        return self.vectors[word]
    
    def clear_cache(self):
        """Clear the embedding cache to free memory"""
        self.vectors = {}

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

            # Average value of the semantic distance between the cue and each word
            # multiplied by 100, which ranges from 0 to 200 
            # (the more distant the higher the score)
            novelty_distances = [self.distance(w1, w2) for w1, w2 in itertools.combinations(subset, 2)]
            novelty = float((sum(novelty_distances) / len(novelty_distances)) * 100)

            dissimilarity_distances = [self.distance(cue, w1) for w1 in subset]
            dissimilarity = float(sum(dissimilarity_distances) / len(dissimilarity_distances) * 100)
            appropriateness = 200 - dissimilarity

        return ((novelty, appropriateness), num_unique)
