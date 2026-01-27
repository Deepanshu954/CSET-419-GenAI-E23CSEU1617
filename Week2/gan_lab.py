import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# --- Configuration ---
# You can change these or override them via input if needed, 
# but for this script we'll set defaults and allow user input as per instructions.

import argparse

def get_config():
    parser = argparse.ArgumentParser(description='GAN Lab Training')
    parser.add_argument('--dataset', type=str, default='fashion', choices=['mnist', 'fashion'], help='Dataset choice: mnist or fashion')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--noise_dim', type=int, default=100, help='Dimension of random noise vector')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--save_interval', type=int, default=5, help='Save interval for generated images')
    
    args = parser.parse_args()
    
    config = {
        'dataset_choice': args.dataset,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'noise_dim': args.noise_dim,
        'lr': args.lr,
        'save_interval': args.save_interval,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu' 
    }
    
    # Try using MPS if on Mac and CUDA is not available
    if config['device'] == 'cpu' and torch.backends.mps.is_available():
         config['device'] = 'mps'

    print("--- GAN Lab Configuration ---")
    for k, v in config.items():
        print(f"{k}: {v}")
        
    return config

# --- Models ---

class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1 * 28 * 28),
            nn.Tanh() # Output is [-1, 1]
        )

    def forward(self, x):
        return self.main(x).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1 * 28 * 28, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

class Classifier(nn.Module):
    """Simple CNN for classification to evaluate GAN quality."""
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))) # 28->14
        x = self.pool(self.relu(self.conv2(x))) # 14->7
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Helper Functions ---

def get_dataloader(dataset_choice, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1]
    ])
    
    if dataset_choice == 'fashion':
        dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    else:
        dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_classifier(classifier, dataloader, device, epochs=5):
    """Trains the classifier to be used for evaluation."""
    print("\n--- Training Evaluator Classifier ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    classifier.to(device)
    classifier.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(dataloader, desc=f"Classifier Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100 * correct / total})
            
    print("Classifier training complete.\n")
    return classifier

# --- Main Execution ---

def main():
    config = get_config()
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs("generated_samples", exist_ok=True)
    os.makedirs("final_generated_images", exist_ok=True)
    
    # Load Data
    dataloader = get_dataloader(config['dataset_choice'], config['batch_size'])
    
    # Initialize Models
    generator = Generator(config['noise_dim']).to(device)
    discriminator = Discriminator().to(device)
    classifier = Classifier().to(device)
    
    # Train Classifier first (needed for evaluation later)
    # In a real scenario, we might load a pre-trained one, but we'll train one quickly here.
    train_classifier(classifier, dataloader, device, epochs=3) # 3 epochs is enough for basic MNIST/Fashion
    
    # Optimizers & Loss
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    
    # Fixed noise for consistent visualization
    fixed_noise = torch.randn(25, config['noise_dim'], device=device)
    
    print("\n--- Starting GAN Training ---")
    
    for epoch in range(config['epochs']):
        d_loss_total = 0
        g_loss_total = 0
        d_acc_total = 0
        batches = 0
        
        generator.train()
        discriminator.train()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        
        for real_images, _ in pbar:
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Labels
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            
            # --- Train Discriminator ---
            optimizer_d.zero_grad()
            
            # Real images
            outputs_real = discriminator(real_images)
            d_loss_real = criterion(outputs_real, real_labels)
            d_loss_real.backward()
            
            # Fake images
            z = torch.randn(batch_size, config['noise_dim'], device=device)
            fake_images = generator(z)
            outputs_fake = discriminator(fake_images.detach()) # Detach to avoid G gradients
            d_loss_fake = criterion(outputs_fake, fake_labels)
            d_loss_fake.backward()
            
            d_loss = d_loss_real + d_loss_fake
            optimizer_d.step()
            
            # Calculate D accuracy
            predicted_real = (outputs_real > 0.5).float()
            predicted_fake = (outputs_fake < 0.5).float()
            d_acc = (predicted_real.eq(real_labels).sum().item() + predicted_fake.eq(fake_labels).sum().item()) / (2 * batch_size)
            
            # --- Train Generator ---
            optimizer_g.zero_grad()
            
            outputs_fake_for_g = discriminator(fake_images) # Re-compute for G (no detach)
            g_loss = criterion(outputs_fake_for_g, real_labels) # Trick D into thinking they are real
            g_loss.backward()
            optimizer_g.step()
            
            d_loss_total += d_loss.item()
            g_loss_total += g_loss.item()
            d_acc_total += d_acc
            batches += 1
            
            pbar.set_postfix({
                'D_loss': d_loss_total/batches, 
                'G_loss': g_loss_total/batches,
                'D_acc': f"{100*d_acc_total/batches:.2f}%"
            })
            
        # End of epoch logging
        print(f"Epoch {epoch+1}/{config['epochs']} | D_loss: {d_loss_total/batches:.4f} | D_acc: {100*d_acc_total/batches:.2f}% | G_loss: {g_loss_total/batches:.4f}")
        
        # Save samples
        if (epoch + 1) % config['save_interval'] == 0 or epoch == 0:
            with torch.no_grad():
                fake_samples = generator(fixed_noise).cpu()
                # Denormalize: [-1, 1] -> [0, 1]
                fake_samples = (fake_samples + 1) / 2
                save_image(fake_samples, f"generated_samples/epoch_{epoch+1:02d}.png", nrow=5)
                print(f"Saved generated samples to generated_samples/epoch_{epoch+1:02d}.png")

    print("\n--- Training Complete ---")
    
    # --- Final Generation & Evaluation ---
    print("\n--- Generating Final Images & Evaluating ---")
    generator.eval()
    classifier.eval()
    
    with torch.no_grad():
        # Generate 100 images
        z_final = torch.randn(100, config['noise_dim'], device=device)
        final_images = generator(z_final)
        
        # Save final images
        final_images_denorm = (final_images + 1) / 2
        save_image(final_images_denorm, "final_generated_images/final_grid.png", nrow=10)
        
        # Save individual images (optional, but good for inspection)
        for i in range(100):
            save_image(final_images_denorm[i], f"final_generated_images/image_{i:03d}.png")
            
        print("Saved 100 final images to final_generated_images/")
        
        # Predict labels
        outputs = classifier(final_images)
        _, predicted = torch.max(outputs, 1)
        
        # Report distribution
        labels = predicted.cpu().numpy()
        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(unique, counts))
        
        print("\nLabel Distribution of Generated Images:")
        for label, count in distribution.items():
            print(f"Class {label}: {count} images")
            
        # Plot distribution
        plt.figure(figsize=(10, 5))
        plt.bar(distribution.keys(), distribution.values())
        plt.xlabel('Class Label')
        plt.ylabel('Count')
        plt.title('Distribution of Generated Image Classes')
        plt.xticks(list(range(10)))
        plt.savefig('final_generated_images/label_distribution.png')
        print("Saved label distribution plot to final_generated_images/label_distribution.png")

if __name__ == "__main__":
    main()
