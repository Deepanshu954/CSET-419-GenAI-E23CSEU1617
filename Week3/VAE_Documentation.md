# Week 3 Lab: Variational Autoencoder (VAE) Documentation

## 1. Introduction
This document explains the concepts and implementation of the Variational Autoencoder (VAE) used in `vae_lab.py`. The goal of this lab is to understand how VAEs learn latent representations of data (specifically MNIST digits) and generate new, diverse samples.

### What is a VAE?
A **Variational Autoencoder (VAE)** is a type of generative model. Unlike a standard Autoencoder (AE), which learns a fixed latent vector for each input, a VAE learns a **probability distribution** (parameters: mean $\mu$ and variance $\sigma^2$) for the latent space.

*   **Standard Autoencoder**: Input $\to$ Encoder $\to$ Fixed Vector $z$ $\to$ Decoder $\to$ Output
*   **Variational Autoencoder**: Input $\to$ Encoder $\to$ Distribution $N(\mu, \sigma)$ $\to$ Sample $z$ $\to$ Decoder $\to$ Output

This probabilistic approach allows VAEs to generate **new** data by sampling from the learned distribution, making them powerful generative models.

---

## 2. Key Concepts & Implementation

### 2.1 The Encoder
**Concept**: The encoder's job is to map the input image $x$ to the parameters of the latent distribution: the mean ($\mu$) and the log-variance ($\log \sigma^2$). We use log-variance for numerical stability (ensuring variance is always positive).

**Code Implementation (`VAE.encode`)**:
```python
def encode(self, x):
    h1 = F.relu(self.fc1(x))
    # Output distinct vectors for mu and log_variance
    return self.fc_mean(h1), self.fc_logvar(h1)
```

### 2.2 The Reparameterization Trick
**Concept**: To train the network using gradient descent (backpropagation), we need to sample $z$ from the distribution $N(\mu, \sigma)$. However, we cannot backpropagate through a random sampling operation.

**Solution**: The **Reparameterization Trick** expresses the random variable $z$ as a deterministic transformation of the parameters and a fixed source of noise $\epsilon$:
$$z = \mu + \sigma \cdot \epsilon$$
where $\epsilon \sim N(0, 1)$ (standard normal distribution). This moves the randomness to $\epsilon$, allowing gradients to flow through $\mu$ and $\sigma$.

**Code Implementation (`VAE.reparameterize`)**:
```python
def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar) # Convert logvar to std
    eps = torch.randn_like(std)   # Sample random noise epsilon
    return mu + eps * std         # z = mu + sigma * epsilon
```

### 2.3 The Decoder
**Concept**: The decoder takes the sampled latent vector $z$ and reconstructs the data to produce $\hat{x}$. It aims to map the latent space back to the original image space (MNIST digits).

**Code Implementation (`VAE.decode`)**:
```python
def decode(self, z):
    h3 = F.relu(self.fc3(z))
    return torch.sigmoid(self.fc4(h3)) # Sigmoid to output pixel values [0, 1]
```

### 2.4 Loss Function
**Concept**: The VAE loss function minimizes two conflicting objectives:
1.  **Reconstruction Loss**: Measures how well the output matches the input. We use **Binary Cross Entropy (BCE)** since MNIST pixels are normalized to $[0, 1]$.
2.  **KL Divergence (Kullback-Leibler)**: A regularization term that forces the learned latent distribution to be close to a standard normal distribution $N(0, 1)$. This ensures the latent space is continuous and can be sampled from easily.

**Equation**:
$$Loss = BCE(x, \hat{x}) + D_{KL}(N(\mu, \sigma) \| N(0, 1))$$

**Code Implementation (`loss_function`)**:
```python
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # Analytic KL Divergence formula for Gaussian distributions
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
```

---

## 3. Training & Generation

### Training Process
The model is trained for 20 epochs. In each step:
1.  The inputs are passed through the encoder to get $\mu$ and $\log \sigma^2$.
2.  Latent vector $z$ is sampled using the reparameterization trick.
3.  The decoder reconstructs the image from $z$.
4.  The combined loss (Reconstruction + KL) is backpropagated to update weights.

### Sample Generation
Once trained, we can generate **new** digits by ignoring the encoder and directly feeding random noise sampled from a standard normal distribution ($N(0, 1)$) into the decoder.

**Code (`generate_samples`)**:
```python
z = torch.randn(64, LATENT_DIM) # Random noise
sample = model.decode(z)        # Generate new images
```

---

## 4. Summary of Results
*   **Latent Representations**: By setting `LATENT_DIM=2`, we can visualize the 2D latent space. Similar digits (e.g., all 1s or all 0s) cluster together in specific regions of this space.
*   **Reconstruction**: The model learns to compress the digits into just 2 numbers and reconstruct them with reasonable accuracy.
*   **Generation**: The model can "dream" up new digit-like images by traversing the learned latent space.
