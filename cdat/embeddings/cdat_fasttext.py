import re
import itertools
import numpy as np
import scipy.spatial.distance
import nltk
nltk.download('averaged_perceptron_tagger_eng')
from nltk.corpus import wordnet

# Optional imports for FastText
FASTTEXT_ERROR = None
try:
    import fasttext
    # Check if load_model method exists
    if not hasattr(fasttext, 'load_model'):
        FASTTEXT_ERROR = (
            "The 'fasttext' module does not have 'load_model' method. "
            "This usually means the wrong package is installed. "
            "Please uninstall and reinstall the correct package:\n"
            "  pip uninstall fasttext\n"
            "  pip install fasttext\n"
            "Or if using conda:\n"
            "  conda install -c conda-forge fasttext"
        )
        FASTTEXT_AVAILABLE = False
    else:
        FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False
    FASTTEXT_ERROR = "FastText library not installed. Install with: pip install fasttext"


class Model:
    """Create model to compute CDAT using FastText"""

    def __init__(self, model_path=None, model_name='fasttext/cc.en.300.bin'):
        """Load FastText model
        
        Args:
            model_path (str, optional): Path to FastText model file (.bin). 
                                       If None, will try to load from model_name
            model_name (str): FastText model name or path
                - Pre-trained models: e.g., 'cc.en.300.bin' (English, 300 dims)
                - Or path to local .bin file: e.g., '/path/to/model.bin'
                - Common models: 'cc.en.300.bin', 'wiki-news-300d-1M.vec', etc.
        """
        if not FASTTEXT_AVAILABLE:
            error_msg = FASTTEXT_ERROR if FASTTEXT_ERROR else (
                "FastText requires 'fasttext' library. "
                "Install with: pip install fasttext"
            )
            raise ImportError(error_msg)
        
        self.model_name = model_name
        self.vectors = {}
        
        # Load FastText model
        # If model_path is provided, use it; otherwise use model_name
        model_file = model_path if model_path else model_name
        
        try:
            self.model = fasttext.load_model(model_file)
        except AttributeError as e:
            # This shouldn't happen if we checked above, but just in case
            raise ImportError(
                f"FastText 'load_model' method not available. "
                f"This usually means the wrong 'fasttext' package is installed.\n"
                f"Please reinstall: pip uninstall fasttext && pip install fasttext"
            )
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load FastText model from '{model_file}'. "
                f"Error: {str(e)}\n"
                f"Please ensure the model file exists. "
                f"You can download pre-trained models from: "
                f"https://fasttext.cc/docs/en/crawl-vectors.html"
            )

    def validate(self, word):
        """Clean up word and ensure it's a valid noun
        
        FastText can handle OOV (out-of-vocabulary) words using subword n-grams,
        so we don't need to check vocabulary existence or generate multiple candidates.
        We just clean the word and filter for nouns.
        """
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
        """Get word embedding using FastText"""
        if word not in self.vectors:
            # FastText: get word vector (handles OOV words using subword n-grams)
            # get_word_vector returns a numpy array
            embedding = self.model.get_word_vector(word)
            self.vectors[word] = embedding
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
            print("Not enough valid words")
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
