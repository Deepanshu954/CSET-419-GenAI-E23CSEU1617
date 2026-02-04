import pickle
import os

INPUT_FILE = 'dataset.txt'
VOCAB_FILE = 'vocab.pkl'

def load_data(seq_length):
    try:
        with open(INPUT_FILE, 'r') as f:
            text = f.read().lower()
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return None, None, None, None, None

    # Check for existing vocab
    if os.path.exists(VOCAB_FILE):
        with open(VOCAB_FILE, 'rb') as f:
            vocab_data = pickle.load(f)
            char_to_int = vocab_data['char_to_int']
            int_to_char = vocab_data['int_to_char']
            chars = vocab_data['chars']
    else:
        chars = sorted(list(set(text)))
        char_to_int = {c: i for i, c in enumerate(chars)}
        int_to_char = {i: c for i, c in enumerate(chars)}
        with open(VOCAB_FILE, 'wb') as f:
            pickle.dump({
                'char_to_int': char_to_int,
                'int_to_char': int_to_char,
                'chars': chars
            }, f)
    
    vocab_size = len(chars)
    
    dataX, dataY = [], []
    for i in range(0, len(text) - seq_length, 1):
        seq_in = text[i:i + seq_length]
        seq_out = text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
        
    return text, dataX, dataY, char_to_int, int_to_char, vocab_size

def get_text_only():
    try:
        with open(INPUT_FILE, 'r') as f:
            return f.read().lower()
    except FileNotFoundError:
        return ""
