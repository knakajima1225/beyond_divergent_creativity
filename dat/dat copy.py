import re
import itertools
import numpy
import scipy.spatial.distance
import nltk
nltk.download('averaged_perceptron_tagger_eng')
from nltk.corpus import wordnet

class Model:
    """Create model to compute DAT"""

    def __init__(self, model="../glove.840B.300d.txt", dictionary="../words_glove.txt", pattern="^[a-z][a-z-]*[a-z]$"):
        """Join model and words matching pattern in dictionary"""

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
                #return cand

                # Check if word is a noun
                word_tagged = nltk.pos_tag([cand])[0][1]
                # Added - For multiple possible tags, better to use WordNet
                possible_pos = set()
                for synset in wordnet.synsets(cand):
                    possible_pos.add(synset.pos())

                if word_tagged in ['NN','NNS'] or 'n' in possible_pos:
                    #print(f"Valid word: {cand} (tagged as {word_tagged})")
                    return cand
                else:
                    #print(f"GloVe invalid word: {word} (tags: POS={word_tagged}, WordNet={possible_pos})")
                    continue

        #print(f"No valid word found for {word}")
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
        #print('Number of valid words before distance', len(uniques), uniques)

        # Keep subset of words
        if len(uniques) >= minimum:
            subset = uniques[:minimum]
        else:
            return (None, len(uniques)) # Not enough valid words

        # Compute distances between each pair of words
        distances = []
        for word1, word2 in itertools.combinations(subset, 2):
            dist = self.distance(word1, word2)
            if dist is not None:  # Only add valid distances
                distances.append(dist)
            else:
                print(f"Distance between {word1} and {word2} is none")
        # Check if we have enough valid distances
        if not distances:
            print("No valid distances found between words")
            return (None, len(uniques))
    
        self.distances = distances
        # Compute the DAT score (average semantic distance multiplied by 100)
        score = (sum(distances) / len(distances)) * 100
        
        return (score, len(uniques))

"""
def dat(self, words, minimum=7):
        #Compute DAT score
        uniques = []
        for word in words:
            valid = self.validate(word)
            if valid and valid not in uniques:
                uniques.append(valid)
        num_unique = len(uniques)
        print('Number of valid words:', num_unique, uniques)

        if num_unique < minimum:
            return None
        subset = uniques[:minimum]

        distances = [self.distance(w1, w2) for w1, w2 in itertools.combinations(subset, 2)]
        score = (sum(distances) / len(distances)) * 100
        return (score, num_unique)
"""