"""
config.py

(ADD DESCRIPTION

Author: Lalit Suresh
Date: 25th October 2025

"""
import os

# Image pathing and directories
DATA_ROOT = "data"
PROCESSED_IMGS_FOLDER = "processed"
RAW_IMGS_FOLDER = "raw"
TRAIN_IMGS_FOLDER = "train-image"

METADATA_CSV_PATH = os.path.join(DATA_ROOT, RAW_IMGS_FOLDER, "train-metadata.csv")
RAW_IMG_DIR = os.path.join(DATA_ROOT, RAW_IMGS_FOLDER, TRAIN_IMGS_FOLDER)
PARTIONED_IMGS_DIR = os.path.join(DATA_ROOT, PROCESSED_IMGS_FOLDER)
CHECKPOINT_PATH = "checkpoints/best_model.pth"
METRICS_CSV = "metrics/epoch_metrics.csv"
SAVE_DIR = "metrics"


# Dataset parameters
BATCH_SIZE = 32
WORKERS = 4
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 20
EMBEDDING_DIM = 256