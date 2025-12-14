import torch
import numpy as np
from sklearn.model_selection import train_test_split
from dataTransformor import DataTransformor

class TensorTransformer(DataTransformor):
    """Transforms data into PyTorch tensors with normalization and splitting."""
    
    def __init__(self, normalize=True, testDataSize=0.2, random_state=42):
        self.normalize = normalize
        self.testDataSize = testDataSize
        self.random_state = random_state
        self.feature_min = None
        self.feature_max = None
    
    def __normalize_features(self, features_tensor):
        """Apply min-max normalization: (x - min) / (max - min)."""
        self.feature_min = features_tensor.min(dim=0, keepdim=True).values
        self.feature_max = features_tensor.max(dim=0, keepdim=True).values
        
        # Avoid division by zero
        denominator = self.feature_max - self.feature_min
        denominator[denominator == 0] = 1
        
        return (features_tensor - self.feature_min) / denominator
    
    def transform(self, features, labels, reshape_labels=True):
        """Transform numpy arrays to PyTorch tensors.
        
        Args:
            features: List or numpy array of features
            labels: List or numpy array of labels
            reshape_labels: Reshape labels to (n_samples, 1) if True
            
        Returns:
            trainFeaturesTensor, testFeaturesTensor, trainLabelsTensor, testLabelsTensor
        """
        # Convert to numpy arrays for cleaning
        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        
        # Clean data: replace NaN and Inf
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        
        # Normalize if requested
        if self.normalize:
            features_tensor = self.__normalize_features(features_tensor)
        
        # Reshape labels
        if reshape_labels:
            labels_tensor = labels_tensor.view(-1, 1)
        
        # Split train/test
        trainFeaturesTensor, testFeaturesTensor, trainLabelsTensor, testLabelsTensor = train_test_split(
            features_tensor, labels_tensor, 
            test_size=self.testDataSize, 
            random_state=self.random_state,
            shuffle=True
        )
        
        return trainFeaturesTensor, testFeaturesTensor, trainLabelsTensor, testLabelsTensor
    
    def normalize_new_data(self, features):
        """Normalize new data using previously computed min/max.
        
        Args:
            features: New feature data to normalize
            
        Returns:
            Normalized features
        """
        if self.feature_min is None or self.feature_max is None:
            raise ValueError("Must call transform() first to compute normalization parameters")
        
        features_tensor = torch.tensor(features, dtype=torch.float32)
        denominator = self.feature_max - self.feature_min
        denominator[denominator == 0] = 1
        
        return (features_tensor - self.feature_min) / denominator