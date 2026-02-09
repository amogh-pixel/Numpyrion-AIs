import numpy as np
import re
from spellchecker import SpellChecker

class Embed_tool():
    # Class Variables (Shared by everything)
    vocab = {
        "a": 0.038, "b": 0.076, "c": 0.115, "d": 0.153, "e": 0.192,
        "f": 0.230, "g": 0.269, "h": 0.307, "i": 0.346, "j": 0.384,
        "k": 0.423, "l": 0.461, "m": 0.500, "n": 0.538, "o": 0.576,
        "p": 0.615, "q": 0.653, "r": 0.692, "s": 0.730, "t": 0.769,
        "u": 0.807, "v": 0.846, "w": 0.884, "x": 0.923, "y": 0.961,
        "z": 1.000, " ": 0.000
    }
    
    # We create the reverse map once here
    reverse_vocab = {v: k for k, v in vocab.items()}

    @staticmethod
    def encode(text, max_len=10):
        # FIX: Use Embed_tool.vocab to access the class variable
        encoded = [Embed_tool.vocab.get(char.lower(), 0.0) for char in text[:max_len]]
        while len(encoded) < max_len:
            encoded.append(0.0)
        return np.array(encoded)

    @staticmethod
    def decode(array):
        decoded = ""
        # FIX: Access the class reverse_vocab
        rev_v = Embed_tool.reverse_vocab
        for val in array:
            # This finds the key with the smallest absolute difference to the AI's output
            closest_val = min(rev_v.keys(), key=lambda x: abs(x - val))
            decoded += rev_v[closest_val]
        return decoded.strip()
    @staticmethod
    def clean(text):
        spell = SpellChecker()
        
        # 1. Standardize (Lowercase & remove symbols)
        text = text.lower()
        words = re.findall(r'\w+', text) # Splits "hello!!!" into ["hello"]
        
        # 2. Find the "bad" words
        misspelled = spell.unknown(words)
        
        clean_words = []
        for word in words:
            if word in misspelled:
                # Get the most likely correct version
                corrected = spell.correction(word)
                # If it can't find a fix, keep the original to avoid crashing
                clean_words.append(corrected if corrected else word)
            else:
                clean_words.append(word)
        
        return " ".join(clean_words)