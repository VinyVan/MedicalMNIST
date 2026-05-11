## Summary

Initial implementation of Medical MNIST classification pipeline with support for 6 medical imaging classes.

## Changes

- Complete project structure with config, src, data, outputs directories
- Support for CNN, ResNet18/34, and EfficientNet-B0 models
- Cross-validation training with early stopping
- Data augmentation for medical images
- Kaggle integration with .env credential loading
- EDA notebook for data exploration

## Features

- 6-class classification (AbdomenCT, BreastMRI, CXR, ChestCT, Hand, HeadCT)
- 64x64 grayscale image processing
- Stratified K-fold cross-validation
- Model checkpointing and OOF predictions
- Automatic data download from Kaggle

## Testing

Run with debug mode:
```bash
python main.py --running_mode debug --model cnn
```

## Dataset

Medical MNIST from Kaggle: https://www.kaggle.com/datasets/andrewmvd/medical-mnist
