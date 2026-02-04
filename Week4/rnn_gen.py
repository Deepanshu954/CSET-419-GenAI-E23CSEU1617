import torch
import torch.nn as nn
import torch.optim as optim
import os
import utils

MODEL_FILE = 'rnn_model.pth'
SEQ_LENGTH = 40
HIDDEN_SIZE = 128
NUM_LAYERS = 1

class TextGeneratorRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(TextGeneratorRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden):
        embed = self.embedding(x)
        out, hidden = self.rnn(embed, hidden)
        out = self.fc(out[:, -1, :]) 
        return out, hidden

def train_model():
    text, dataX, dataY, char_to_int, _, vocab_size = utils.load_data(SEQ_LENGTH)
    if text is None: return

    X = torch.tensor(dataX, dtype=torch.long)
    y = torch.tensor(dataY, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available(): device = torch.device('mps')

    model = TextGeneratorRNN(vocab_size, HIDDEN_SIZE, NUM_LAYERS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    print("Starting Training (Vanilla RNN)...")
    for epoch in range(100):
        model.train()
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            hidden = None 
            output, _ = model(batch_x, hidden)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/100, Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), MODEL_FILE)
    print(f"RNN model saved to {MODEL_FILE}")

def generate_text_model(start_str, length=200):
    if not os.path.exists(MODEL_FILE):
        print("Model not found. Please train first.")
        return ""
        
    _, _, _, char_to_int, int_to_char, vocab_size = utils.load_data(SEQ_LENGTH)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available(): device = torch.device('mps')

    model = TextGeneratorRNN(vocab_size, HIDDEN_SIZE, NUM_LAYERS).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()

    current_string = start_str.lower()
    generated_text = start_str
    
    with torch.no_grad():
        for _ in range(length):
            if len(current_string) >= SEQ_LENGTH: input_seq = current_string[-SEQ_LENGTH:]
            else: input_seq = current_string
            
            input_indices = [char_to_int.get(c, 0) for c in input_seq]
            input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)
            
            output, _ = model(input_tensor, None)
            probs = torch.softmax(output, dim=1)
            next_char_idx = torch.multinomial(probs, 1).item()
            
            next_char = int_to_char[next_char_idx]
            generated_text += next_char
            current_string += next_char
            
    return generated_text

if __name__ == "__main__":
    train_model()
