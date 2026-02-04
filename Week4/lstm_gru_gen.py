import torch
import torch.nn as nn
import torch.optim as optim
import os
import utils

SEQ_LENGTH = 40
HIDDEN_SIZE = 256
NUM_LAYERS = 2

class TextGeneratorLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(TextGeneratorLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden):
        embed = self.embedding(x)
        out, hidden = self.lstm(embed, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

class TextGeneratorGRU(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(TextGeneratorGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden):
        embed = self.embedding(x)
        out, hidden = self.gru(embed, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

def train_model(model_type='LSTM'):
    text, dataX, dataY, _, _, vocab_size = utils.load_data(SEQ_LENGTH)
    if text is None: return

    X = torch.tensor(dataX, dtype=torch.long)
    y = torch.tensor(dataY, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available(): device = torch.device('mps')

    if model_type == 'LSTM':
        model = TextGeneratorLSTM(vocab_size, HIDDEN_SIZE, NUM_LAYERS).to(device)
        model_file = 'lstm_model.pth'
    else:
        model = TextGeneratorGRU(vocab_size, HIDDEN_SIZE, NUM_LAYERS).to(device)
        model_file = 'gru_model.pth'

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    print(f"Starting Training ({model_type})...")
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

    torch.save(model.state_dict(), model_file)
    print(f"{model_type} model saved to {model_file}")

def generate_text_model(start_str, model_type='LSTM', length=200):
    model_file = 'lstm_model.pth' if model_type == 'LSTM' else 'gru_model.pth'
    
    if not os.path.exists(model_file):
        print(f"Model {model_file} not found. Please train first.")
        return ""
        
    _, _, _, char_to_int, int_to_char, vocab_size = utils.load_data(SEQ_LENGTH)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available(): device = torch.device('mps')

    if model_type == 'LSTM':
        model = TextGeneratorLSTM(vocab_size, HIDDEN_SIZE, NUM_LAYERS).to(device)
    else:
        model = TextGeneratorGRU(vocab_size, HIDDEN_SIZE, NUM_LAYERS).to(device)
        
    model.load_state_dict(torch.load(model_file, map_location=device))
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
    train_model('LSTM')
