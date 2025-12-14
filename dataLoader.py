from abc import ABC, abstractmethod

class DataLoader(ABC):
    """Abstract base class for all data loaders."""
    
    @abstractmethod
    def loadData(self, filePath: str):
        """Load data from specified file path.
        
        Args:
            filePath: Path to data file
            
        Returns:
            Tuple of (features, labels)
        """
        pass