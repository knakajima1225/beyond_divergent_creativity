import re
import itertools
import numpy as np
import scipy.spatial.distance
from sentence_transformers import SentenceTransformer
import nltk
nltk.download('averaged_perceptron_tagger_eng')
from nltk.corpus import wordnet
from nltk.corpus import words

class Model:
    """Create model to compute DAT using BERT"""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Load BERT-based model"""
        self.model = SentenceTransformer(model_name)
        self.vectors = {}
        # Load English dictionary for word validation
        self.english_words = set(words.words())

    def validate(self, word):
        """Clean up word and ensure it's valid"""
        clean = re.sub(r"[^a-zA-Z- ]+", "", word).strip().lower()
        
        if len(clean) > 1:
            # First check if it's a real English word
            if clean not in self.english_words:
                #print(f"BERT invalid word (not in dictionary): {word}")
                return None
                
            word_tagged = nltk.pos_tag([clean])[0][1]
            #print(f"Word tagged: {word}, {word_tagged}")

            # Added on 23/04/2025 - For multiple possible tags, better to use WordNet
            possible_pos = set()
            for synset in wordnet.synsets(clean):
                possible_pos.add(synset.pos())
                

            if word_tagged in ['NN','NNS'] or 'n' in possible_pos:
                return clean
            else:
                #print(f"BERT invalid word: {word} (tags: POS={word_tagged}, WordNet={possible_pos})")
                return None
        else:
            return None

    def embed(self, word):
        """Get word embedding"""
        if word not in self.vectors:
            self.vectors[word] = self.model.encode(word, convert_to_numpy=True)
        return self.vectors[word]

    def distance(self, word1, word2):
        """Compute cosine distance (0 to 2) between two words"""
        vec1, vec2 = self.embed(word1), self.embed(word2)
        return scipy.spatial.distance.cosine(vec1, vec2)

    def dat(self, words, minimum=7):
        """Compute DAT score"""
        uniques = []
        for word in words:
            valid = self.validate(word)
            if valid and valid not in uniques:
                uniques.append(valid)
        num_unique = len(uniques)
        #print('Number of valid words:', num_unique)

        if num_unique < minimum:
            print("Score is None")
            return (None, num_unique)
        subset = uniques[:minimum]

        distances = [self.distance(w1, w2) for w1, w2 in itertools.combinations(subset, 2)]
        score = (sum(distances) / len(distances)) * 100
        return (score, num_unique)
