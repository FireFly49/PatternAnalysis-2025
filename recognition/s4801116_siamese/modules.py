"""
modules.py

Initialises the neural network
modules for the binary classifier
and Siamese network

Author: Lalit Suresh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SiameseNetwork(nn.Module):
    """
    Siamese network to use images
    to generate embeddings
    """
    def __init__(self, embedding_dim=256):
        """
        Siamese network initialiser with ResNet34
        backbone.

        Args: 
            embedding_dim (int): Dimensions for embedding output.
        """
        super(SiameseNetwork, self).__init__()

        # Backbone
        self.backbone = models.resnet34(weights='ResNet34_Weights.DEFAULT')
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Improved embedding head
        self.embedding_head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward_single(self, x):
        """
        Forward pass for a single image batch. Passes input images
        through the ResNet34 backbone and 
        embedding head to produce normalized embeddings.

        Args:
            x (torch.Tensor): Batch of input images of shape [B, 3, H, W].

        Returns:
            torch.Tensor: L2-normalized embeddings of shape [B, embedding_dim].
        """
        features = self.backbone(x)
        embedding = self.embedding_head(features)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    def forward(self, anchor, positive, negative):
        """
        Forward pass for a triplet of images. Computes embeddings
        for the anchor, positive, and negative samples
        in a triplet used for the triplet loss function.

        Args:
            anchor (torch.Tensor): Batch of anchor images.
            positive (torch.Tensor): Batch of positive (same class) images.
            negative (torch.Tensor): Batch of negative (different class) images.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                Normalized embeddings for (anchor, positive, negative).
        """
        anchor_emb = self.forward_single(anchor)
        pos_emb = self.forward_single(positive)
        neg_emb = self.forward_single(negative)
        return anchor_emb, pos_emb, neg_emb
    
    def get_embeddings(self, x):
        """
        Computes embeddings for a single batch of images without gradient tracking.

        Used during validation or inference to generate 
        fixed-length feature embeddings from input images.

        Args:
            x (torch.Tensor): Batch of input images of shape [B, 3, H, W].

        Returns:
            torch.Tensor: L2-normalized embeddings of shape [B, embedding_dim].
        """
        self.eval()
        with torch.no_grad():
            features = self.backbone(x)
            embeddings = self.embedding_head(features)
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
        


class LesionClassifier(nn.Module):
    """
    Simple binary classifier that takes a precomputed embedding
    (e.g., from a pretrained Siamese network) and predicts 0/1.
    """
    def __init__(self, embedding_dim=256, hidden_dim=128):
        """
        Classifier that takes in an embedding and outputs binary 
        class targets for lesion classification.

        Args:
            embedding_dim (int): Dimension of input embeddings.
            hidden_dim (int): Dimension of hidden layer.
        
        """ 
        super(LesionClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, embedding):
        """
        Predicts binary class from input embedding.

        Args:
            embedding (torch.Tensor): Input embedding tensor of shape [batch_size, embedding_dim].
        
        Returns:
            prediction (torch.Tensor): Output logits of shape [batch_size, 2].

        """
        prediction = self.classifier(embedding)
        return prediction