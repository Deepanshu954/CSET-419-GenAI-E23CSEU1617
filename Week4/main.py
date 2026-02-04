import sys
import os

# Import model modules
import ngram_gen
import rnn_gen
import lstm_gru_gen
import transformer_gen

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_menu():
    print("\n--- Text Generation Lab ---")
    print("1. N-gram Model")
    print("2. RNN Model")
    print("3. LSTM Model")
    print("4. GRU Model")
    print("5. Transformer Model")
    print("0. Exit")
    return input("Select an option: ")

def train_or_generate():
    print("\n1. Train Model (This will overwrite existing model)")
    print("2. Generate Text (Uses saved model)")
    print("0. Back")
    return input("Select an option: ")

def handle_ngram():
    while True:
        choice = train_or_generate()
        if choice == '1':
            ngram_gen.train_model()
        elif choice == '2':
            seed = input("Enter seed text (first word): ")
            print("\n--- Output ---")
            print(ngram_gen.generate_text_model(seed))
        elif choice == '0':
            break

def handle_rnn():
    while True:
        choice = train_or_generate()
        if choice == '1':
            rnn_gen.train_model()
        elif choice == '2':
            seed = input("Enter seed text (first word): ")
            print("\n--- Output ---")
            print(rnn_gen.generate_text_model(seed))
        elif choice == '0':
            break

def handle_lstm():
    while True:
        choice = train_or_generate()
        if choice == '1':
            lstm_gru_gen.train_model('LSTM')
        elif choice == '2':
            seed = input("Enter seed text (first word): ")
            print("\n--- Output ---")
            print(lstm_gru_gen.generate_text_model(seed, 'LSTM'))
        elif choice == '0':
            break

def handle_gru():
    while True:
        choice = train_or_generate()
        if choice == '1':
            lstm_gru_gen.train_model('GRU')
        elif choice == '2':
            seed = input("Enter seed text (first word): ")
            print("\n--- Output ---")
            print(lstm_gru_gen.generate_text_model(seed, 'GRU'))
        elif choice == '0':
            break

def handle_transformer():
    while True:
        choice = train_or_generate()
        if choice == '1':
            transformer_gen.train_model()
        elif choice == '2':
            seed = input("Enter seed text (first word): ")
            print("\n--- Output ---")
            print(transformer_gen.generate_text_model(seed))
        elif choice == '0':
            break

def main():
    while True:
        choice = print_menu()
        
        if choice == '1':
            handle_ngram()
        elif choice == '2':
            handle_rnn()
        elif choice == '3':
            handle_lstm()
        elif choice == '4':
            handle_gru()
        elif choice == '5':
            handle_transformer()
        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
