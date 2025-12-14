from abc import ABC, abstractmethod

class DataTransformor(ABC):
    """Abstract base class for data transformation."""
    
    @abstractmethod
    def transform(self, features, labels, reshape_labels=True):
        """Transform raw features and labels.
        
        Args:
            features: Raw feature data
            labels: Raw label data
            reshape_labels: Whether to reshape labels tensor
            
        Returns:
            Transformed data suitable for model training
        """
        pass