# Skin Lesion Analysis Towards Melanoma Detection using a Siamese Network

## Overview

## Environment setup

Miniconda was used for setting up a virtual environment for this solution for Python 3.12.9. However this setup
process may also be replicated by using anaconda for the same effect. The setup process is as 
follows:

1. Install [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) or [anaconda](https://www.anaconda.com/download)
2. Create a virtual environment of a chosen name using the following ('envname' is a placeholder)

    ```
    conda create -n envname python=3.12.9
    conda activate envname
    ```

3. Install the following packages were used in this project. You may use pip or conda to do so
    - PyTorch: 2.7.1 (With CUDA 11.8 support)
    - torchvision: 0.22.1
    - scikit-learn: 1.6.1
    - matplotlib: 3.10.3
    - seaborn: 0.13.2
    - pandas: 2.2.3
    - numpy: 2.2.6

## Data preprocessing

No explicit test dataset was provided for this solution, so the resized training set
from Kaggle was repurposed and split into training, testing and validation sets
in proportions of (tt split proportions)

## Network Architecure

### Siamese Network 


### Loss function

## Model Training

### Parameters

Model training was implemented in `train.py`
Run the following script to begin the training process

```
python train.py
```

## Model Evaluation

Graphs made for analyis will be stored under the `graphs` folder after running the `predict.py` script

## Results

### Confusion Matrix


## Conclusion