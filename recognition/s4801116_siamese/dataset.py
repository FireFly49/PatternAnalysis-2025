import os
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

TRAIN_SPLIT = 0.8
TEST_SPLIT = 0.2