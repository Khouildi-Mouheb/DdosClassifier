import torch
import torch.nn as nn

class ClassificationModel(nn.Module):
    """Binary classification model for DDoS detection."""
    
    def __init__(self, numberOfFeatures=0):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define network architecture
        self.layerstack = nn.Sequential(
            nn.Linear(numberOfFeatures, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Move model to device immediately
        self.to(self.device)
        
        # Loss and optimizer
        self.lossFn = nn.BCEWithLogitsLoss()
        self.learningRate = 0.001
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate)
        
        # Track training metrics
        self.training_history = {
            'train_loss': [],
            'test_loss': [],
            'train_acc': [],
            'test_acc': []
        }
    
    def forward(self, X: torch.Tensor):
        """Forward pass through the network.
        
        Args:
            X: Input tensor of shape (batch_size, n_features)
            
        Returns:
            Raw logits (before sigmoid)
        """
        # Ensure input is on correct device
        if X.device != self.device:
            X = X.to(self.device)
        return self.layerstack(X)
    
    def predict_proba(self, X: torch.Tensor):
        """Get probability predictions.
        
        Args:
            X: Input tensor
            
        Returns:
            Probabilities between 0 and 1
        """
        self.eval()
        with torch.no_grad():
            logits = self(X)
            probabilities = torch.sigmoid(logits)
        return probabilities
    
    def predict(self, X: torch.Tensor, threshold=0.5):
        """Get binary predictions.
        
        Args:
            X: Input tensor
            threshold: Decision threshold
            
        Returns:
            Binary predictions (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities > threshold).float()
    
    def trainModel(self, 
                   TrainFeaturesTensor, 
                   TestFeaturesTensor,
                   trainLabelsTensor, 
                   testLabelsTensor,
                   numberOfEpochs=1000,
                   print_every=100):
        """Train the model.
        
        Args:
            TrainFeaturesTensor: Training features
            TestFeaturesTensor: Testing features
            trainLabelsTensor: Training labels
            testLabelsTensor: Testing labels
            numberOfEpochs: Number of training epochs
            print_every: Print metrics every N epochs
        """
        # Move data to correct device
        TrainFeaturesTensor = TrainFeaturesTensor.to(self.device)
        TestFeaturesTensor = TestFeaturesTensor.to(self.device)
        trainLabelsTensor = trainLabelsTensor.to(self.device)
        testLabelsTensor = testLabelsTensor.to(self.device)
        
        print(f"Training on {self.device}")
        print(f"Training samples: {len(TrainFeaturesTensor)}")
        print(f"Testing samples: {len(TestFeaturesTensor)}")
        
        for epoch in range(numberOfEpochs):
            self.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            train_logits = self(TrainFeaturesTensor)
            train_loss = self.lossFn(train_logits, trainLabelsTensor)
            
            # Backward pass
            train_loss.backward()
            self.optimizer.step()
            
            # Evaluate
            if epoch % print_every == 0 or epoch == numberOfEpochs - 1:
                self.eval()
                with torch.no_grad():
                    # Training metrics
                    train_probs = torch.sigmoid(train_logits)
                    train_preds = (train_probs > 0.5).float()
                    train_acc = (train_preds == trainLabelsTensor).float().mean() * 100
                    
                    # Testing metrics
                    test_logits = self(TestFeaturesTensor)
                    test_loss = self.lossFn(test_logits, testLabelsTensor)
                    test_probs = torch.sigmoid(test_logits)
                    test_preds = (test_probs > 0.5).float()
                    test_acc = (test_preds == testLabelsTensor).float().mean() * 100
                    
                    # Store history
                    self.training_history['train_loss'].append(train_loss.item())
                    self.training_history['test_loss'].append(test_loss.item())
                    self.training_history['train_acc'].append(train_acc.item())
                    self.training_history['test_acc'].append(test_acc.item())
                    
                    print(f"Epoch {epoch:4d}/{numberOfEpochs} | "
                          f"Train Loss: {train_loss.item():.4f} | "
                          f"Test Loss: {test_loss.item():.4f} | "
                          f"Train Acc: {train_acc:.2f}% | "
                          f"Test Acc: {test_acc:.2f}%")
        
        print("Training completed.")
        return self.training_history
    
    def save_model(self, filepath):
        """Save model weights.
        
        Args:
            filepath: Path to save model weights
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model weights.
        
        Args:
            filepath: Path to load model weights from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        print(f"Model loaded from {filepath}")