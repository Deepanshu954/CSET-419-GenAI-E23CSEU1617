# GenAI Lab - Week 2: Basic GAN Implementation

This project implements a basic Generative Adversarial Network (GAN) to generate synthetic images resembling the MNIST or Fashion-MNIST datasets. It includes a Generator, a Discriminator, and a Classifier to evaluate the quality of the generated images.

## Requirements

- Python 3.x
- PyTorch
- Torchvision
- Matplotlib
- Numpy
- Tqdm

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:

```bash
python gan_lab.py
```

You will be prompted to enter the following configuration parameters (defaults are provided):

- **Dataset Choice**: `mnist` or `fashion` (Default: `mnist`)
- **Epochs**: Number of training epochs (Default: `50`)
- **Batch Size**: Batch size for training (Default: `64`)
- **Noise Dimension**: Size of the random noise vector (Default: `100`)
- **Learning Rate**: Learning rate for optimizers (Default: `0.0002`)
- **Save Interval**: Interval (in epochs) to save generated sample grids (Default: `5`)

## Output

The program generates the following outputs:

1.  **Training Logs**: Printed to the console, showing D_loss, G_loss, and D_acc per epoch.
2.  **Generated Samples**: Saved in the `generated_samples/` directory.
    - `epoch_XX.png`: Grid of generated images at specific epochs.
3.  **Final Generated Images**: Saved in the `final_generated_images/` directory.
    - `final_grid.png`: A 10x10 grid of 100 final generated images.
    - `image_XXX.png`: Individual generated images.
    - `label_distribution.png`: A bar chart showing the predicted class distribution of the generated images.
4.  **Label Distribution**: Printed to the console at the end of training.

## Models

- **Generator**: A fully connected network that maps noise vectors to 28x28 images.
- **Discriminator**: A fully connected network that classifies images as real or fake.
- **Classifier**: A simple CNN trained on the dataset to evaluate the generated images (Transfer Learning concept).
