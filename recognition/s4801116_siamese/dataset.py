"""
dataset.py

Generates dataset partitions
and dataloaders for the ISIC2020 
dataset

Author: Lalit Suresh
"""

import os
import random
import shutil
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data.sampler import WeightedRandomSampler # To generate Balanced batches
from torchvision import transforms

from sklearn.model_selection import train_test_split
from collections import Counter

from tqdm import tqdm
from config import BATCH_SIZE, WORKERS, MEAN, STD, PARTIONED_IMGS_DIR, METADATA_CSV_PATH, RAW_IMG_DIR

# Credit to https://github.com/TheNoZER0/PatternAnalysis-2024/tree/topic-recognition/recognition/Siamese-48008361
# for base structure of dataset file

class ISIC2020Dataset(Dataset):
    """
    Custom PyTorch Dataset for ISIC 2020 Kaggle challenge
    Returns (anchor, positive, negative) triplet samples
    """
    BENIGN_LABEL = 0
    MALIGNANT_LABEL = 1

    def __init__(self, data_dir, mode, transform=None):
        """
        Initialise dataset by supplying path for
        dataset images.

        Args:
            data_dir (str): Path to the root directory containing 'benign' and 'malignant' folders.
            mode (str): One of 'train' or 'val' to specify dataset split.
            transform (torchvision.transforms): Transformations to apply to images.
        """
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        self.benign_dir = os.path.join(data_dir, 'benign')
        self.malignant_dir = os.path.join(data_dir, 'malignant')

        benign_paths_labels = self.get_img_paths(self.BENIGN_LABEL)
        malignant_paths_labels = self.get_img_paths(self.MALIGNANT_LABEL)
        
        all_imgs_labels = benign_paths_labels + malignant_paths_labels
        
        all_paths = [img for img, _ in all_imgs_labels]
        all_labels = [label for _, label in all_imgs_labels]

        train_paths, val_paths, train_labels, val_labels = train_test_split(
            all_paths, 
            all_labels, 
            test_size=0.2, 
            random_state=42, 
            stratify=all_labels
        )

        if self.mode == 'train':
            self.images = list(zip(train_paths, train_labels))
        elif self.mode == 'val':
            self.images = list(zip(val_paths, val_labels))
        else:
            raise ValueError("Mode must be 'train' or 'val'")

        self.indices_by_target = {
            self.BENIGN_LABEL: [],
            self.MALIGNANT_LABEL: [],
        }

        for idx, (_, target) in enumerate(self.images):
            self.indices_by_target[target].append(idx)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            length (int): Number of samples.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns a triplet (anchor, positive, negative) for the given index.

        Args:
            idx (int): Index of the anchor image.
        
        Returns:
            anchor_img (torch.Tensor): Anchor image tensor.
            positive_img (torch.Tensor): Positive image tensor.
            negative_img (torch.Tensor): Negative image tensor.
            anchor_target (int): Target label of the anchor image.
        """
        anchor_path, anchor_target = self.images[idx]
        
        possible_pos_indices = self.indices_by_target[anchor_target]
        pos_idx = idx
        while pos_idx == idx:
            pos_idx = random.choice(possible_pos_indices)
        positive_path, _ = self.images[pos_idx]

        negative_target = self.MALIGNANT_LABEL if anchor_target == self.BENIGN_LABEL else self.BENIGN_LABEL
        possible_neg_indices = self.indices_by_target[negative_target]
        neg_idx = random.choice(possible_neg_indices)
        negative_path, _ = self.images[neg_idx]

        anchor_img = self.load_image(anchor_path)
        positive_img = self.load_image(positive_path)
        negative_img = self.load_image(negative_path)

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        return anchor_img, positive_img, negative_img, anchor_target
    
    def calculate_sampler_weights(self):
        """
        Calculates the inverse-frequency weights for each sample 
        in the current dataset (self.images). This should be called
        on the train_dataset instance after imblearn has run,
        or on the val_dataset instance.

        Returns:
            sample_weights (list): List of weights for each sample.
        """
        labels = self.all_labels
        counts = Counter(labels)
        total_samples = len(labels)
        
        class_weights = {
            cls: total_samples / count
            for cls, count in counts.items()
        }
        
        sample_weights = [
            class_weights[label]
            for label in labels
        ]
        
        return sample_weights
    
    @property
    def all_labels(self):
        """
        Returns the list of labels in the current dataset split (train or val).
        This list reflects the oversampling applied during initialization for 'train' mode,
        allowing external validation of class counts.

        Returns:
            list of labels for all images (list).
        """
        return [label for _, label in self.images]

    def load_image(self, path):
        """
        Load an image found in the given file path.

        Args:
            path (str): File path to the image.

        Returns:
            image (PIL.Image): Loaded image in RGB format.
        """
        return Image.open(path).convert('RGB')

    def get_img_paths(self, target_label):
        """
        Get paths of all images in the respective class (Benign or malignant).

        Args:
            target_label (int): 0 for benign, 1 for malignant.

        Returns:
            img_paths (list): List of tuples (image_path, target_label).
        """
        img_dir = self.malignant_dir if target_label else self.benign_dir
        img_paths = []
        with os.scandir(img_dir) as files:
            for file in files:
                img_paths.append(file.path) if file.name.endswith('.jpg') else None

        return [(path, target_label) for path in img_paths]



def partition_data(metadata_csv, raw_img_dir, output_dir):
    """
    Processes raw image training set into relevant categories
    using provided metadata into 'benign' and 'malignant'
    folders.

    Args:
        metadata_csv (str): File path to metadata for raw image data.
        raw_img_dir (str): File path to raw image data.
        output_dir (str): Output directory for partitioned dataset.

    """

    # Read metadata file
    metadata = pd.read_csv(metadata_csv)
    metadata = metadata[['isic_id','target']]

    # Create output directories for sorted files
    benign_imgs_out = os.path.join(output_dir, "benign")
    malignant_imgs_out = os.path.join(output_dir, "malignant")

    os.makedirs(benign_imgs_out, exist_ok=True)
    os.makedirs(malignant_imgs_out, exist_ok=True)

    # For each image description in the metadata, find the relevant 
    # image in the raw images and link them together
    # We ignore the participant, since this will not
    # help us for this use case

    def classify_row(row):
        img_id = row['isic_id']
        label = row['target']
        img_name = img_id + ".jpg"
        
        source_path = os.path.join(raw_img_dir, img_name)
        dest_dir = malignant_imgs_out if label == 1 else benign_imgs_out

        dest_path = os.path.join(dest_dir, img_name)

        # File operation (copying the file)
        try:
            shutil.copyfile(source_path, dest_path) # Copy files to new folder
        except FileNotFoundError:
            print(f"Error: Source image not found at {source_path}")
        except Exception as e:
            print(f"Error copying {img_id}: {e}")
    print("\n---------------------------------------------------------")
    print(f"Starting data partitioning for {len(metadata)} images...")
    
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Partitioning Images"):
        classify_row(row)
    
    print("Completed image partitioning")
    print("\n---------------------------------------------------------")


def get_data_loaders(partitioned_imgs_dir=PARTIONED_IMGS_DIR):
    """
    Generate PyTorch custom 
    dataloaders for the ISIC dataset
    for validation and training sets.

    Args:
        partitioned_imgs_dir (str): Path to partitioned images directory.

    Returns:
        train_loader (DataLoader): DataLoader for training set.
        val_loader (DataLoader): DataLoader for validation set.
    """

    # Transforms were borrowed from the following
    # site https://pmc.ncbi.nlm.nih.gov/articles/PMC11766406/

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=20),
        
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    train_set = ISIC2020Dataset(partitioned_imgs_dir, 'train', transform=train_transform)
    val_set = ISIC2020Dataset(partitioned_imgs_dir, 'val', transform=val_transform)

    sample_weights = train_set.calculate_sampler_weights()
    train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_set), replacement=True)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader

def main():
    partition_data(METADATA_CSV_PATH, RAW_IMG_DIR, PARTIONED_IMGS_DIR)

    train_loader, val_loader = get_data_loaders()

    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    
    train_benign_count = sum(1 for label in train_dataset.all_labels if label == 0)
    train_malignant_count = sum(1 for label in train_dataset.all_labels if label == 1)
    val_benign_count = sum(1 for label in val_dataset.all_labels if label == 0)
    val_malignant_count = sum(1 for label in val_dataset.all_labels if label == 1)

    print("\nOverall Dataset Statistics:")
    print(f"Training set - Total: {len(train_dataset)}, Benign: {train_benign_count}, Malignant: {train_malignant_count}")
    print(f"Validation set - Total: {len(val_dataset)}, Benign: {val_benign_count}, Malignant: {val_malignant_count}")



if __name__ == "__main__":
    main()