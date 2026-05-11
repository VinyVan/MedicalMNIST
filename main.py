"""
Main entry point for Medical MNIST Classification Pipeline
Run via: python main.py --running_mode debug --model cnn

Dataset: Medical MNIST (6 classes of medical images)
- AbdomenCT
- BreastMRI
- CXR (Chest X-Ray)
- ChestCT
- Hand
- HeadCT

Image size: 64x64 pixels, Grayscale
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

import torch
import numpy as np
import pandas as pd

from config.config import Config
from config.paths import Paths
from src.data_loader import (
    load_medical_mnist_data,
    create_data_loaders,
    create_cv_splits,
    download_medical_mnist_kaggle,
    check_data_exists,
    split_train_test
)
from src.modeling.models import get_model, print_model_summary, count_parameters
from src.training.trainer import CVTrainer
from src.utils import set_seed, setup_logging, get_device
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    parser = argparse.ArgumentParser(description="Medical MNIST Classification Pipeline")
    
    # Running mode
    parser.add_argument("--running_mode", type=str, default="debug",
                       choices=["debug", "train", "predict"],
                       help="Pipeline running mode")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="cnn",
                       choices=["cnn", "cnn_deep", "resnet18", "resnet34", "efficientnet_b0"],
                       help="Model architecture to use")
    
    # Training parameters
    parser.add_argument("--fold", type=int, default=5,
                       help="Number of cross-validation folds")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of training epochs (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size (overrides config)")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate (overrides config)")
    
    # Cross-validation
    parser.add_argument("--cv_strategy", type=str, default="stratified_kfold",
                       choices=["stratified_kfold", "kfold"],
                       help="Cross-validation strategy")
    
    # Data options
    parser.add_argument("--download", action="store_true", default=False,
                       help="Download dataset from Kaggle if not present")
    parser.add_argument("--test_size", type=float, default=0.2,
                       help="Fraction of data to use as test set")
    
    # Predictions
    parser.add_argument("--submit", action="store_true", default=False,
                       help="Generate submission file")
    
    # Device
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to use for training")
    
    args = parser.parse_args()
    
    # Initialize config
    config = Config()
    config.n_folds = args.fold
    config.cv_strategy = args.cv_strategy
    config.model_name = args.model
    config.device = get_device(args.device)
    
    # Override config with command line args
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    
    # Set seed
    set_seed(config.seed)
    
    # Setup logging
    log_name = f"run_{args.running_mode}_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = setup_logging(log_name)
    
    # Print header
    print("=" * 60)
    print("MEDICAL MNIST CLASSIFICATION PIPELINE")
    print("=" * 60)
    print(f"Mode: {args.running_mode}")
    print(f"Model: {config.model_name}")
    print(f"Device: {config.device}")
    print(f"Folds: {config.n_folds}")
    print(f"CV Strategy: {config.cv_strategy}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print("=" * 60)
    
    logger.info(f"Starting pipeline: mode={args.running_mode}, model={config.model_name}")
    
    # Ensure directories
    Paths.ensure_directories()
    
    # Check for CUDA
    if config.device == "cuda":
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    print("\n1. LOADING DATA")
    print("-" * 40)
    
    # Check if data exists, download if needed
    if not check_data_exists(Paths.RAW_DATA_DIR) or args.download:
        print("Data not found. Downloading from Kaggle...")
        try:
            download_medical_mnist_kaggle(Paths.RAW_DATA_DIR)
        except Exception as e:
            print(f"Error downloading data: {e}")
            print("Please download manually from: https://www.kaggle.com/datasets/andrewmvd/medical-mnist")
            return
    
    # Load data
    image_paths, labels, class_names = load_medical_mnist_data(Paths.RAW_DATA_DIR)
    
    # Update config class names
    config.class_names = class_names
    config.num_classes = len(class_names)
    
    print(f"\nTotal images: {len(image_paths)}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    
    # Log label distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nLabel distribution:")
    for cls_idx, count in zip(unique, counts):
        print(f"  {class_names[cls_idx]}: {count}")
    
    # Debug mode: use smaller sample
    if args.running_mode == "debug":
        print(f"\n[DEBUG MODE] Using {config.debug_sample_size} samples")
        image_paths = image_paths[:config.debug_sample_size]
        labels = labels[:config.debug_sample_size]
    
    # Split into train/test
    train_paths, train_labels, test_paths, test_labels = split_train_test(
        image_paths, labels, test_size=args.test_size, random_state=config.seed
    )
    
    print(f"\nTrain samples: {len(train_paths)}")
    print(f"Test samples: {len(test_paths)}")
    
    print("\n2. MODEL CONFIGURATION")
    print("-" * 40)
    
    # Create model
    model = get_model(config.model_name, config)
    print_model_summary(model)
    logger.info(f"Model: {config.model_name}, Parameters: {count_parameters(model):,}")
    
    print("\n3. CROSS-VALIDATION TRAINING")
    print("-" * 40)
    
    # Create CV splits
    cv_splits = create_cv_splits(train_paths, train_labels, config)
    
    # Storage for OOF predictions
    oof_predictions = np.zeros(len(train_paths))
    oof_probabilities = np.zeros((len(train_paths), config.num_classes))
    fold_scores = []
    trained_models = []
    
    # Train each fold
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\nFold {fold_idx + 1}/{config.n_folds}")
        print("-" * 40)
        
        # Get fold data
        fold_train_paths = [train_paths[i] for i in train_idx]
        fold_train_labels = [train_labels[i] for i in train_idx]
        fold_val_paths = [train_paths[i] for i in val_idx]
        fold_val_labels = [train_labels[i] for i in val_idx]
        
        # Create data loaders
        train_loader, val_loader, _ = create_data_loaders(
            fold_train_paths, fold_train_labels,
            fold_val_paths, fold_val_labels,
            None,
            config
        )
        
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # Create fresh model for this fold
        fold_model = get_model(config.model_name, config)
        
        # Create trainer
        trainer = CVTrainer(fold_model, config)
        
        # Train
        history = trainer.fit(train_loader, val_loader, fold_idx)
        fold_scores.append(history['best_val_acc'])
        
        # Store OOF predictions
        _, _, fold_preds = trainer.validate(val_loader)
        oof_predictions[val_idx] = fold_preds
        
        # Save fold model
        trainer.save_model(Paths.MODELS_DIR, fold=fold_idx)
        trained_models.append(trainer)
        
        logger.info(f"Fold {fold_idx+1} complete - Best Val Acc: {history['best_val_acc']:.2f}%")
    
    # Calculate CV score
    cv_accuracy = 100. * accuracy_score(train_labels, oof_predictions)
    
    print(f"\n{'='*60}")
    print(f"Cross-Validation Results")
    print(f"{'='*60}")
    print(f"Fold scores: {[f'{s:.2f}%' for s in fold_scores]}")
    print(f"Mean CV Accuracy: {np.mean(fold_scores):.2f}% (+/- {np.std(fold_scores):.2f}%)")
    print(f"OOF Accuracy: {cv_accuracy:.2f}%")
    print(f"{'='*60}")
    
    # Detailed classification report
    print("\nClassification Report (OOF):")
    print(classification_report(train_labels, oof_predictions, target_names=class_names, labels=list(range(config.num_classes))))
    
    logger.info(f"CV complete - Mean Acc: {np.mean(fold_scores):.2f}%, OOF Acc: {cv_accuracy:.2f}%")
    
    # Save OOF predictions
    if args.running_mode != "debug":
        oof_df = pd.DataFrame({
            'image_path': [str(p) for p in train_paths],
            'oof_prediction': oof_predictions.astype(int),
            'true_label': train_labels
        })
        oof_df.to_csv(Paths.OOF_DIR / 'oof_predictions.csv', index=False)
        logger.info(f"OOF predictions saved to {Paths.OOF_DIR / 'oof_predictions.csv'}")
    
    # Test predictions
    if len(test_paths) > 0:
        print("\n4. EVALUATING ON TEST SET")
        print("-" * 40)
        
        # Create test data loader
        _, _, test_loader = create_data_loaders(
            train_paths[:1], [0], test_paths, test_labels, config
        )
        
        # Ensemble predictions from all folds
        all_predictions = []
        all_probabilities = []
        
        for fold_idx, trainer in enumerate(trained_models):
            print(f"Predicting with fold {fold_idx + 1} model...")
            predictions, probabilities = trainer.predict(test_loader)
            all_predictions.append(predictions)
            all_probabilities.append(probabilities)
        
        # Average predictions
        avg_probabilities = np.mean(all_probabilities, axis=0)
        final_predictions = np.argmax(avg_probabilities, axis=1)
        
        test_accuracy = 100. * accuracy_score(test_labels, final_predictions)
        print(f"\nTest Accuracy: {test_accuracy:.2f}%")
        
        print("\nTest Set Classification Report:")
        print(classification_report(test_labels, final_predictions, target_names=class_names, labels=list(range(config.num_classes))))
        
        logger.info(f"Test Accuracy: {test_accuracy:.2f}%")
        
        # Generate submission if requested
        if args.submit:
            print("\n5. GENERATING SUBMISSION")
            print("-" * 40)
            
            submission_path = Paths.SUBMISSIONS_DIR / f"submission_{config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            submission_df = pd.DataFrame({
                'image_path': [str(p) for p in test_paths],
                'predicted_label': final_predictions,
                'true_label': test_labels
            })
            submission_df.to_csv(submission_path, index=False)
            
            # Log prediction distribution
            unique, counts = np.unique(final_predictions, return_counts=True)
            pred_dist = {class_names[u]: c for u, c in zip(unique, counts)}
            logger.info(f"Submission distribution: {pred_dist}")
            logger.info(f"Submission saved to {submission_path}")
            print(f"Submission saved to {submission_path}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Mean CV Accuracy: {np.mean(fold_scores):.2f}%")
    print(f"OOF Accuracy: {cv_accuracy:.2f}%")
    if len(test_paths) > 0:
        print(f"Test Accuracy: {test_accuracy:.2f}%")
    print("=" * 60)
    
    logger.info("Pipeline complete successfully")


if __name__ == "__main__":
    main()
