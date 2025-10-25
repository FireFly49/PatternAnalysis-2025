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

from tqdm import tqdm

# Hyperparameters
BATCH_SIZE = 32
WORKERS = 4
TRAIN_SPLIT = 0.7
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1

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
    def __init__(self, data_path, mode, transform):
        """
        Initialise dataset by supplying path for
        dataset images

        Args:
            - data_path (str): Directory pointing to relevant images
            - mode (str): 'train' for training set, 'test' for testing set
            - transform (callable, optional): Image transforms to be 
            - applied on specified data 
        """

        pass

    def __len__(self):
        """
        Returns the total number of samples in the dataset
        """


    def __getitem__(self, idx):
        """
        Fetch a triplet sample (anchor, positive, negative)
        """
        pass

    def load_image(self, path):
        """
        Load an image found in the given 
        file path
        """
        pass




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
    partition_data(metadata_csv_path, raw_img_dir, partitioned_imgs_dir)