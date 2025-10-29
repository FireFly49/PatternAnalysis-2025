"""
modules.py

(ADD DESCRIPTION)

Author: Lalit Suresh
Date: 25th October 2025

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SiameseNetwork(nn.Module):
    """
    A Siamese Network architecture using a ResNet-34 backbone for 
    deep metric learning with Triplet Loss. 
    
    The network generates a low-dimensional embedding (256D) for each input image.
    """

    def __init__(self, embedding_dim=256):
        """
        Initializes the Siamese Network structure.

        Args:
            embedding_dim (int): The final dimension of the embedding vector.
        """
        super(SiameseNetwork, self).__init__()

        # --- 1. Backbone: Use pre-trained ResNet-34 ---
        # Note: Set pretrained=True to use ImageNet weights
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        # Get the size of the features before the final FC layer (512 for ResNet-34)
        num_ftrs = self.backbone.fc.in_features
        
        # --- 2. Feature Extractor Modification ---
        # Remove the final classification layer
        self.backbone.fc = nn.Identity()

        # --- 3. Embedding Head ---
        # Add a custom head to project the features into the desired embedding space
        self.embedding_head = nn.Sequential(
            # Apply dropout for regularization
            nn.Dropout(0.5), 
            # Final linear layer to map 512 features to the embedding dimension (256)
            nn.Linear(num_ftrs, embedding_dim),
            # L2 Normalization ensures the embeddings live on a hypersphere,
            # which is crucial for distance-based metric learning.
            nn.BatchNorm1d(embedding_dim) 
        )

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes a single input image through the network to produce its embedding.

        Args:
            x (torch.Tensor): The input image (e.g., Anchor, Positive, or Negative).

        Returns:
            torch.Tensor: The normalized 256D embedding vector.
        """
        # Pass image through the modified ResNet backbone
        features = self.backbone(x)
        
        # Pass features through the embedding head
        embedding = self.embedding_head(features)
        
        # L2 normalize the embedding (important for distance metrics)
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding

    def forward(self, anchor, positive, negative):
        """
        Processes a triplet (Anchor, Positive, Negative) simultaneously.
        
        Args:
            anchor (torch.Tensor): The anchor batch.
            positive (torch.Tensor): The positive batch.
            negative (torch.Tensor): The negative batch.

        Returns:
            tuple: (anchor_embedding, positive_embedding, negative_embedding)
        """
        # Process each item in the triplet
        anchor_embed = self.forward_single(anchor)
        positive_embed = self.forward_single(positive)
        negative_embed = self.forward_single(negative)
        
        return anchor_embed, positive_embed, negative_embed
