import re
import itertools
import numpy as np
import scipy.spatial.distance
from transformers import AutoTokenizer, AutoModel
import torch
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet

class Model:
    """BERT-only model to compute embeddings from layers 3-9 and CDAT scores per layer"""

    def __init__(self, model_name='bert-base-uncased', device=None):
        """Initialize BERT model"""
        self.model_name = model_name
        self.vectors = {}  # cache: {word: {layer: vector}}
        
        # Device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        # Load model & tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.to(self.device)
        self.model.eval()

        # layers to extract (3 to 9 inclusive)
        self.layers = list(range(3, 10))

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
        """Get embeddings for layers 3-9 individually and average across layers"""
        if word in self.vectors:
            return self.vectors[word]

        with torch.no_grad():
            inputs = self.tokenizer(word, return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k,v in inputs.items()}
            outputs = self.model(**inputs)
            hidden_states = outputs.hidden_states  # tuple of layers (0-12)
            
            embeddings_per_layer = {}
            for layer_idx in self.layers:
                layer_hidden = hidden_states[layer_idx]  # [batch, seq_len, hidden_size]
                attention_mask = inputs['attention_mask']
                mask_expanded = attention_mask.unsqueeze(-1).expand(layer_hidden.size()).float()
                sum_embeddings = torch.sum(layer_hidden * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = (sum_embeddings / sum_mask).squeeze().cpu().numpy()
                embeddings_per_layer[layer_idx] = pooled

            # average across layers
            avg_embedding = np.mean(list(embeddings_per_layer.values()), axis=0)
            embeddings_per_layer['avg'] = avg_embedding

            self.vectors[word] = embeddings_per_layer
            return embeddings_per_layer

    def distance(self, vec1, vec2):
        """Compute cosine distance (0 to 2) between two words"""
        return scipy.spatial.distance.cosine(vec1, vec2)

    def cdat(self, cue, words, minimum=7):
        """Compute CDAT score per layer and average"""
        uniques = []
        for word in words:
            valid = self.validate(word)
            if valid and valid not in uniques:
                uniques.append(valid)

        num_unique = len(uniques)
        if num_unique < minimum:
            return ({layer: (None, None) for layer in self.layers + ['avg']}, num_unique)
        
        subset = uniques[:minimum]

        # Compute embeddings for cue and words
        cue_embeds = self.embed(cue)
        word_embeds_list = [self.embed(w) for w in subset]

        results = {}
        for layer in self.layers + ['avg']:
            # Novelty: pairwise distances between words
            novelty_dists = [self.distance(word_embeds_list[i][layer], word_embeds_list[j][layer])
                             for i,j in itertools.combinations(range(minimum), 2)]
            novelty = float((sum(novelty_dists)/len(novelty_dists))*100)

            # Dissimilarity: cue to each word
            dissim_dists = [self.distance(cue_embeds[layer], word_embeds_list[i][layer])
                             for i in range(minimum)]
            dissimilarity = float(sum(dissim_dists)/len(dissim_dists)*100)
            appropriateness = 200 - dissimilarity

            results[layer] = (novelty, appropriateness)

        return results, num_unique

    def clear_cache(self):
        """Clear the embedding cache to free memory"""
        self.vectors = {}
