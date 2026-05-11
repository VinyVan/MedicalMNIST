# Medical MNIST Classification Pipeline

A deep learning pipeline for classifying medical images from the [Medical MNIST dataset](https://www.kaggle.com/datasets/andrewmvd/medical-mnist).

## Dataset

**Medical MNIST** contains 6 classes of medical images:
- **AbdomenCT** - Abdominal CT scans
- **BreastMRI** - Breast MRI images
- **CXR** - Chest X-Ray images
- **ChestCT** - Chest CT scans
- **Hand** - Hand X-rays
- **HeadCT** - Head CT scans

- **Image size**: 64x64 pixels
- **Format**: Grayscale (JPEG)
- **Total images**: ~58,000 images (approximately 10,000 per class)

## Project Structure

```
MedicalMNIST/
├── config/
│   ├── config.py          # Centralized configuration
│   └── paths.py           # Path management
├── src/
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── modeling/
│   │   └── models.py        # Model architectures
│   ├── training/
│   │   └── trainer.py       # Training loop and CV
│   └── utils.py           # Utility functions
├── data/
│   ├── raw/               # Downloaded dataset
│   └── processed/         # Processed data
├── outputs/
│   ├── models/            # Saved model weights
│   ├── oof/               # Out-of-fold predictions
│   ├── logs/              # Training logs
│   └── submissions/       # Submission files
├── eda/                   # Exploratory data analysis
├── notebooks/             # Jupyter notebooks
├── main.py               # Main entry point
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Data

The pipeline can automatically download the dataset from Kaggle:

```bash
python main.py --download
```

Or download manually from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/medical-mnist) and extract to `data/raw/`.

### 3. Run the Pipeline

**Debug mode** (small sample, fast):
```bash
python main.py --running_mode debug --model cnn
```

**Full training**:
```bash
python main.py --running_mode train --model cnn --fold 5
```

**With transfer learning**:
```bash
python main.py --running_mode train --model resnet18 --fold 5
```

## Usage

### Command Line Arguments

```bash
python main.py [options]

Options:
  --running_mode {debug,train,predict}  Pipeline mode (default: debug)
  --model {cnn,cnn_deep,resnet18,resnet34,efficientnet_b0}
                                        Model architecture (default: cnn)
  --fold N                             Number of CV folds (default: 5)
  --epochs N                           Number of epochs (overrides config)
  --batch_size N                       Batch size (overrides config)
  --lr FLOAT                           Learning rate (overrides config)
  --cv_strategy {stratified_kfold,kfold}
                                        CV strategy (default: stratified_kfold)
  --download                           Download dataset from Kaggle
  --test_size FLOAT                    Fraction for test set (default: 0.2)
  --submit                             Generate submission file
  --device {auto,cuda,cpu}             Device for training (default: auto)
```

### Examples

```bash
# Debug with custom CNN
python main.py --running_mode debug --model cnn --epochs 5

# Train with deeper CNN, 5-fold CV
python main.py --running_mode train --model cnn_deep --fold 5

# Train ResNet with custom hyperparameters
python main.py --running_mode train --model resnet18 --epochs 30 --batch_size 128 --lr 0.0001

# Use CPU only
python main.py --running_mode debug --model cnn --device cpu
```

## Model Architectures

### 1. SimpleCNN
Custom CNN designed for 64x64 medical images:
- 3 convolutional blocks
- Batch normalization and dropout
- ~500K parameters
- Fast training

### 2. DeeperCNN
Deeper version with more capacity:
- 4 convolutional blocks
- More feature maps
- ~2M parameters
- Better for complex patterns

### 3. ResNet18/34
Transfer learning from ImageNet:
- Pre-trained weights
- Requires RGB conversion (grayscale → 3 channels)
- Strong baseline performance

### 4. EfficientNet-B0
State-of-the-art architecture:
- Compound scaling
- Efficient parameter usage
- Best accuracy/speed tradeoff

## Training Features

- **Cross-Validation**: Stratified K-Fold to ensure balanced classes
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Plateau-based reduction
- **Data Augmentation**: Rotation, translation, scaling, brightness/contrast
- **OOF Predictions**: Out-of-fold predictions for ensemble analysis
- **Model Checkpointing**: Saves best model for each fold

## Configuration

Edit `config/config.py` to modify:
- Hyperparameters (learning rate, batch size, epochs)
- Image size and channels
- Augmentation parameters
- Model architecture options

## Results

The pipeline outputs:
- **Cross-validation accuracy**: Mean accuracy across all folds
- **OOF predictions**: Out-of-fold predictions for error analysis
- **Test accuracy**: Performance on held-out test set
- **Classification report**: Per-class precision, recall, F1-score
- **Training logs**: Detailed logs in `outputs/logs/`

## License

This project is for educational purposes. The Medical MNIST dataset is available on Kaggle under its respective license.

## Acknowledgments

- Dataset: [Medical MNIST on Kaggle](https://www.kaggle.com/datasets/andrewmvd/medical-mnist)
- Based on the original [MedNIST dataset](https://github.com/apolanco3225/MedNIST)
