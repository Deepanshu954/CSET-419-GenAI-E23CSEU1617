import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
import utils

MODEL_FILE = 'transformer_model.pth'
SEQ_LENGTH = 40
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TextGeneratorTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(TextGeneratorTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
    
    def forward(self, x):
        embed = self.embedding(x) * math.sqrt(self.d_model)
        src = self.pos_encoder(embed)
        output = self.transformer_encoder(src)
        out = self.fc(output[:, -1, :]) 
        return out

def train_model():
    text, dataX, dataY, _, _, vocab_size = utils.load_data(SEQ_LENGTH)
    if text is None: return

    X = torch.tensor(dataX, dtype=torch.long)
    y = torch.tensor(dataY, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available(): device = torch.device('mps')

    model = TextGeneratorTransformer(vocab_size, D_MODEL, NHEAD, NUM_LAYERS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    print("Starting Training (Transformer)...")
    for epoch in range(100):
        model.train()
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/100, Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), MODEL_FILE)
    print(f"Transformer model saved to {MODEL_FILE}")

def generate_text_model(start_str, length=200):
    if not os.path.exists(MODEL_FILE):
        print("Model not found. Please train first.")
        return ""
        
    _, _, _, char_to_int, int_to_char, vocab_size = utils.load_data(SEQ_LENGTH)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available(): device = torch.device('mps')

    model = TextGeneratorTransformer(vocab_size, D_MODEL, NHEAD, NUM_LAYERS).to(device)
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
            
            output = model(input_tensor)
            
            probs = torch.softmax(output, dim=1)
            next_char_idx = torch.multinomial(probs, 1).item()
            
            next_char = int_to_char[next_char_idx]
            generated_text += next_char
            current_string += next_char
            
    return generated_text

if __name__ == "__main__":
    train_model()
