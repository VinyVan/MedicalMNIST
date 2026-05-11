"""
Data loading and preprocessing for Medical MNIST Classification
Supports image folder structure (class folders with images)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Callable, List
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from PIL import Image
import logging
import os
from config.config import Config

logger = logging.getLogger(__name__)


def _load_kaggle_credentials_from_env():
    """Load Kaggle credentials from .env file in parent directories"""
    try:
        from dotenv import load_dotenv
        
        # Search for .env in current and parent directories
        current_dir = Path(__file__).resolve().parent
        search_dirs = [
            current_dir,
            current_dir.parent,
            current_dir.parent.parent,
            current_dir.parent.parent.parent,
        ]
        
        for search_dir in search_dirs:
            env_file = search_dir / '.env'
            if env_file.exists():
                load_dotenv(env_file)
                logger.info(f"Loaded credentials from: {env_file}")
                break
        
        # Set Kaggle environment variables
        kaggle_username = os.getenv('KAGGLE_USERNAME')
        kaggle_key = os.getenv('KAGGLE_KEY')
        
        if kaggle_username and kaggle_key:
            os.environ['KAGGLE_USERNAME'] = kaggle_username
            os.environ['KAGGLE_KEY'] = kaggle_key
            logger.info("Kaggle credentials loaded from .env")
            return True
        else:
            logger.warning("Kaggle credentials not found in .env")
            return False
            
    except ImportError:
        logger.warning("python-dotenv not installed, cannot load .env file")
        return False
    except Exception as e:
        logger.warning(f"Could not load .env file: {e}")
        return False


class MedicalMNISTDataset(Dataset):
    """Dataset for Medical MNIST image folder structure"""
    
    def __init__(
        self, 
        image_paths: List[Path],
        labels: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        is_train: bool = True
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_train = is_train
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image (Medical MNIST is grayscale)
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        return image, idx


class ImageFolderDataset(Dataset):
    """Generic Dataset for image folder structure (class folders)"""
    
    def __init__(
        self,
        image_paths: list,
        labels: Optional[list] = None,
        transform: Optional[Callable] = None,
        convert_rgb: bool = False
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.convert_rgb = convert_rgb
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Convert to RGB if needed (for transfer learning models)
        mode = 'RGB' if self.convert_rgb else 'L'
        image = Image.open(img_path).convert(mode)
        
        if self.transform:
            image = self.transform(image)
        
        if self.labels is not None:
            return image, self.labels[idx]
        return image, idx


def get_train_transforms(config, is_train: bool = True):
    """Get image transforms for training/validation"""
    transforms_list = []
    
    # Safety check for config
    if config is None:
        # Default transforms if no config provided
        transforms_list.extend([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        return transforms.Compose(transforms_list)
    
    # For transfer learning models, we may need RGB
    if config.num_channels == 3:
        transforms_list.append(transforms.Lambda(lambda x: x.convert('RGB') if x.mode == 'L' else x))
    
    # Convert to tensor first
    transforms_list.append(transforms.ToTensor())
    
    if is_train and config.use_augmentation:
        aug_params = config.augmentation_params
        transforms_list.extend([
            transforms.RandomRotation(aug_params.get("rotation", 15)),
            transforms.RandomAffine(
                degrees=0,
                translate=(aug_params.get("translation", 0.1),) * 2,
                scale=aug_params.get("scale", (0.9, 1.1)),
                shear=aug_params.get("shear", 5)
            ),
            transforms.ColorJitter(
                brightness=aug_params.get("brightness", 0.2),
                contrast=aug_params.get("contrast", 0.2)
            ),
        ])
    
    # Normalize
    transforms_list.append(
        transforms.Normalize(
            mean=config.normalize_mean,
            std=config.normalize_std
        )
    )
    
    return transforms.Compose(transforms_list)


def load_medical_mnist_data(data_dir: Path) -> Tuple[List[Path], List[int], List[str]]:
    """
    Load Medical MNIST data from folder structure
    
    Expected structure:
        data_dir/
            AbdomenCT/
                image1.jpeg
                image2.jpeg
                ...
            BreastMRI/
            CXR/
            ChestCT/
            Hand/
            HeadCT/
    
    Returns:
        image_paths: List of image file paths
        labels: List of integer labels
        class_names: List of class names
    """
    data_dir = Path(data_dir)
    
    # Define class mapping (alphabetical order)
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    
    image_paths = []
    labels = []
    
    logger.info(f"Loading Medical MNIST data from {data_dir}")
    logger.info(f"Found classes: {class_names}")
    
    for class_name in class_names:
        class_dir = data_dir / class_name
        class_images = list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        
        image_paths.extend(class_images)
        labels.extend([class_to_idx[class_name]] * len(class_images))
        
        logger.info(f"  {class_name}: {len(class_images)} images")
    
    logger.info(f"Total images: {len(image_paths)}")
    
    return image_paths, labels, class_names


def create_data_loaders(
    train_paths: List[Path],
    train_labels: List[int],
    val_paths: Optional[List[Path]] = None,
    val_labels: Optional[List[int]] = None,
    test_paths: Optional[List[Path]] = None,
    config: Optional[Config] = None
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """Create PyTorch DataLoaders for train/val/test"""
    
    # Get transforms
    train_transform = get_train_transforms(config, is_train=True)
    val_transform = get_train_transforms(config, is_train=False)
    
    # Create datasets
    train_dataset = ImageFolderDataset(
        train_paths, 
        train_labels, 
        transform=train_transform,
        convert_rgb=(config.num_channels == 3)
    )
    
    val_dataset = None
    if val_paths is not None and val_labels is not None:
        val_dataset = ImageFolderDataset(
            val_paths, 
            val_labels, 
            transform=val_transform,
            convert_rgb=(config.num_channels == 3)
        )
    
    test_dataset = None
    if test_paths is not None:
        test_dataset = ImageFolderDataset(
            test_paths, 
            None, 
            transform=val_transform,
            convert_rgb=(config.num_channels == 3)
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device == "cuda" else False
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True if config.device == "cuda" else False
        )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True if config.device == "cuda" else False
        )
    
    return train_loader, val_loader, test_loader


def create_cv_splits(
    image_paths: List[Path],
    labels: List[int],
    config,
    n_splits: Optional[int] = None
):
    """Create cross-validation splits for image classification"""
    n_splits = n_splits or config.n_folds
    
    labels_array = np.array(labels)
    
    if config.cv_strategy == "stratified_kfold":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.seed)
        splits = list(cv.split(np.zeros(len(labels)), labels_array))
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=config.seed)
        splits = list(cv.split(range(len(labels))))
    
    return splits


def split_train_test(
    image_paths: List[Path],
    labels: List[int],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[List[Path], List[int], List[Path], List[int]]:
    """Split data into train and test sets"""
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    return train_paths, train_labels, test_paths, test_labels


def download_medical_mnist_kaggle(output_dir: Path):
    """Download Medical MNIST dataset from Kaggle"""
    # Load credentials from .env file first
    _load_kaggle_credentials_from_env()
    
    try:
        import kaggle
        logger.info("Downloading Medical MNIST dataset from Kaggle...")
        logger.info("Dataset: andrewmvd/medical-mnist")
        
        # Download and unzip
        kaggle.api.dataset_download_files(
            'andrewmvd/medical-mnist',
            path=output_dir,
            unzip=True,
            quiet=False
        )
        logger.info(f"Downloaded and extracted to {output_dir}")
        
        # Check the extracted structure
        extracted_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
        logger.info(f"Extracted directories: {[d.name for d in extracted_dirs]}")
        
    except Exception as e:
        logger.error(f"Failed to download Medical MNIST: {e}")
        logger.info("Please download manually from: https://www.kaggle.com/datasets/andrewmvd/medical-mnist")
        raise


def check_data_exists(data_dir: Path) -> bool:
    """Check if Medical MNIST data exists in the expected structure"""
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        return False
    
    # Check for at least some of the expected class directories
    expected_classes = ["AbdomenCT", "BreastMRI", "CXR", "ChestCT", "Hand", "HeadCT"]
    found_classes = [d.name for d in data_dir.iterdir() if d.is_dir()]
    
    return len(set(expected_classes) & set(found_classes)) >= 4
