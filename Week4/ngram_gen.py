import random
import pickle
import os
from collections import defaultdict
import utils

MODEL_FILE = 'ngram_model.pkl'

def train_model():
    text = utils.get_text_only()
    if not text:
        return None
        
    n = 3
    ngrams = defaultdict(lambda: defaultdict(int))

    for i in range(len(text) - n):
        gram = text[i:i + n - 1]
        next_char = text[i + n - 1]
        ngrams[gram][next_char] += 1

    # Convert to dict for pickling (defaultdict lambda fails pickle)
    model = {}
    for gram, next_chars in ngrams.items():
        total_count = sum(next_chars.values())
        model[gram] = {char: count / total_count for char, count in next_chars.items()}
    
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"N-gram model saved to {MODEL_FILE}")
    return model

def generate_text_model(start_str, length=200):
    if not os.path.exists(MODEL_FILE):
        print("Model not found. Please train first.")
        return ""
    
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
        
    n = 3
    start_str = start_str.lower()
    generated_text = start_str
    
    # Needs at least n-1 chars
    if len(start_str) < n - 1:
        start_str = start_str.rjust(n-1, ' ') # Pad if needed

    current_gram = start_str[-(n-1):]
    
    for _ in range(length):
        if current_gram not in model:
             # Random fallback is tricky without full text access, but we can assume ' ' or stop
             # Just break for safety or loop a char?
             # Let's just break if we can't predict
             break 
        else:
            next_chars_probs = model[current_gram]
            chars = list(next_chars_probs.keys())
            probs = list(next_chars_probs.values())
            next_char = random.choices(chars, weights=probs, k=1)[0]
        
        generated_text += next_char
        current_gram = generated_text[-(n-1):]
        
    return generated_text

if __name__ == "__main__":
    train_model()
    print(generate_text_model("artificial"))
