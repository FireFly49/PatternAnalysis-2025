"""
dataset.py

(ADD DESCRIPTION)

Author: Lalit Suresh
Date: 25th October 2025

"""

import os
import random
import shutil
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data.sampler import WeightedRandomSampler # To generate Balanced batches
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

from tqdm import tqdm

# Hyperparameters
BATCH_SIZE = 32
WORKERS = 4
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

def preprocess_data():
    pass

class ISIC2020Dataset(Dataset):
    """
    Custom PyTorch Dataset for ISIC 2020 Kaggle challenge

    Terminology:
        - anchor: image with a given class
        - positive: image with same class as anchor
        - negative: image with opposite class to anchor

    Returns (anchor, positive, negative) triplet samples
    """
    BENIGN_LABEL = 0
    MALIGNANT_LABEL = 1

    def __init__(self, data_dir, mode, transform=None):
        """
        Initialise dataset by supplying path for
        dataset images

        Args:
            - data_dir (str): Directory pointing to relevant images
            - mode (str): 'train' for training set, 'test' for testing set
            - transform (callable, optional): Image transforms to be 
            - applied on specified data 
        """

        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        self.benign_dir = os.path.join(data_dir, 'benign')
        self.malignant_dir = os.path.join(data_dir, 'malignant')

        # Define benign and malignant images
        self.benign_paths = self.get_img_paths(self.BENIGN_LABEL)
        self.malignant_paths = self.get_img_paths(self.MALIGNANT_LABEL)

        all_imgs_labels = self.benign_paths + self.malignant_paths
        self.all_imgs = [img for img, _ in all_imgs_labels]
        self.all_labels = [label for _, label in all_imgs_labels]

        # 2. Perform stratified split (using 80/20 split as default)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            self.all_imgs, 
            self.all_labels, 
            test_size=0.2, 
            random_state=42, 
            stratify=self.all_labels
        )

        if self.mode == 'train':
            self.images = list(zip(train_paths, train_labels))
        elif self.mode == 'val':
            self.images = list(zip(val_paths, val_labels))
        else:
            raise ValueError("Mode must be 'train' or 'val'")

        # 4. Create indices dictionary for fast triplet sampling
        self.indices_by_target = {
            self.BENIGN_LABEL: [],
            self.MALIGNANT_LABEL: [],
        }

        for idx, (_, target) in enumerate(self.images):
            self.indices_by_target[target].append(idx)

    def __len__(self):
        """
        Returns the total number of samples in the dataset
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns a triplet of samples 
        (anchor, positive, negative)

        """
        # --- 1. Get Anchor (A) ---
        anchor_path, anchor_target = self.images[idx]
        
        # --- 2. Get Positive (P) ---
        # Find all indices of the same target class (excluding the anchor's own index)
        possible_pos_indices = self.indices_by_target[anchor_target]
        
        # Ensure we don't pick the anchor itself as the positive
        pos_idx = idx
        while pos_idx == idx:
            pos_idx = random.choice(possible_pos_indices)
            
        positive_path, _ = self.images[pos_idx]

        # --- 3. Get Negative (N) ---
        # The negative target is the opposite class
        negative_target = self.MALIGNANT_LABEL if anchor_target == self.BENIGN_LABEL else self.BENIGN_LABEL
        possible_neg_indices = self.indices_by_target[negative_target]
        
        # Randomly choose one index from the opposite class
        neg_idx = random.choice(possible_neg_indices)
        negative_path, _ = self.images[neg_idx]

        # --- 4. Load and Transform Images ---
        anchor_img = self.load_image(anchor_path)
        positive_img = self.load_image(positive_path)
        negative_img = self.load_image(negative_path)

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        # Return the triplet (A, P, N) and the original anchor label (for sanity check/debugging)
        return anchor_img, positive_img, negative_img, anchor_target

    def load_image(self, path):
        """
        Load an image found in the given 
        file path
        """
        return Image.open(path).convert('RGB')

    def get_img_paths(self, target_label):
        """
        Get paths of all images in the respective
        class (Benign or malignant)

        Args:
            - target_label (int): 1 or 0 to indicate malignant or benign lesions

        Returns:
            - pass
        """

        img_dir = self.malignant_dir if target_label else self.benign_dir
        img_paths = os.listdir(img_dir)
        return list(map(lambda x: (os.path.join(img_dir, x), target_label), img_paths))
    
    def oversample_minority(self):
        """
        Performs oversampling of 
        the minority class 'malignant'
        by randomly duplicating entries
        using RandomOverSampler
        from imblearn library

        Args:
            - oversample_ratio (float): Ratio to oversample 
            
        """

        # Reshape paths array to allow the arrays
        # to work with RandomOverSampler
        X = np.array(self.all_imgs).reshape(-1, 1) 
        y = np.array(self.all_labels)

        oversampler = RandomOverSampler(random_state=42)
        X_over, y_over = oversampler.fit_resample(X, y)

        oversampled_paths = X_over.ravel().tolist()
        oversampled_labels = y_over.tolist()
        
        return oversampled_paths, oversampled_labels
        



def partition_data(metadata_csv, raw_img_dir, output_dir):
    """
    Processes raw image training set into relevant categories
    using provided metadata into 'benign' and 'malignant'
    folders

    Args:
        - metadata_csv (str): File path to metadata for raw image data
        - raw_img_dir (str): File path to raw image data
        - output_dir (str): Output directory for partitioned dataset

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
    
    # --- PROGRESS BAR IMPLEMENTATION ---
    # We iterate over the rows using iterrows() and wrap it with tqdm()
    # total=len(metadata) is crucial for accurate time estimates
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Partitioning Images"):
        classify_row(row)
    
    print("Completed image partitioning 😊")
    print("\n---------------------------------------------------------")


def get_data_loaders():
    pass

def main():
    pass



if __name__ == "__main__":
    
    # CHANGE ABSOLUTE PATH TO RELATIVE PATH LATER
    metadata_csv_path = r"C:\Users\lalit\Documents\Uni\yr_3\sem_2\comp_3710\PatternAnalysis-2025\recognition\s4801116_siamese\data\raw\train-metadata.csv"
    raw_img_dir = r"C:\Users\lalit\Documents\Uni\yr_3\sem_2\comp_3710\PatternAnalysis-2025\recognition\s4801116_siamese\data\raw\train-image"
    partitioned_imgs_dir = r"C:\Users\lalit\Documents\Uni\yr_3\sem_2\comp_3710\PatternAnalysis-2025\recognition\s4801116_siamese\data\processed"
    # partition_data(metadata_csv_path, raw_img_dir, partitioned_imgs_dir)
    train_set = ISIC2020Dataset(partitioned_imgs_dir, 'train')
    test_set = ISIC2020Dataset(partitioned_imgs_dir, 'val')

    train_set.oversample_minority()

    # anchor, positive, negative, target = train_set[100] 
    # anchor_label = "Malignant" if target == ISIC2020Dataset.MALIGNANT_LABEL else "Benign"

    # # Use PIL's built-in show() method to display the image object
    # print(f"Displaying raw PIL Image object (Target: {anchor_label}).")
    # anchor.show()