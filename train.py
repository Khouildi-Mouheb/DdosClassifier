import torch
from csv_data_loader import CSVDataLoader
from tensorTransformer import TensorTransformer
from classificationModel import ClassificationModel
import pickle

def train():
    """Main training function."""
    print("Starting DDoS Classifier Training...")
    
    # 1. Load data
    print("Step 1: Loading CSV data...")
    loader = CSVDataLoader(
        r"CSVdata\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        hasHeader=True
    )
    
    try:
        features, labels = loader.loadData()
        print(f"  ✓ Loaded {len(features)} samples")
        print(f"  ✓ Number of features: {loader.getNumberOfFeatures()}")
        print(f"  ✓ Label mapping: {loader.getLabelsMapping()}")
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        return
    
    # 2. Transform data
    print("Step 2: Transforming data...")
    transformer = TensorTransformer(normalize=True, testDataSize=0.2)
    
    try:
        trainFeatureTensor, testFeatureTensor, trainLabelTensor, testLabelTensor = transformer.transform(features, labels)
        
        print(f"  ✓ Train features shape: {trainFeatureTensor.shape}")
        print(f"  ✓ Test features shape: {testFeatureTensor.shape}")
        print(f"  ✓ Train labels shape: {trainLabelTensor.shape}")
        print(f"  ✓ Test labels shape: {testLabelTensor.shape}")
        
        # Check class distribution
        train_pos = (trainLabelTensor == 1).sum().item()
        train_neg = (trainLabelTensor == 0).sum().item()
        test_pos = (testLabelTensor == 1).sum().item()
        test_neg = (testLabelTensor == 0).sum().item()
        
        print(f"  ✓ Train class balance: {train_neg} normal, {train_pos} attack")
        print(f"  ✓ Test class balance: {test_neg} normal, {test_pos} attack")
        
    except Exception as e:
        print(f"  ✗ Error transforming data: {e}")
        return
    
    # 3. Create and train model
    print("Step 3: Training model...")
    n_features = trainFeatureTensor.shape[1]
    model = ClassificationModel(numberOfFeatures=n_features)
    
    try:
        history = model.trainModel(
            TrainFeaturesTensor=trainFeatureTensor,
            TestFeaturesTensor=testFeatureTensor,
            trainLabelsTensor=trainLabelTensor,
            testLabelsTensor=testLabelTensor,
            numberOfEpochs=200,
            print_every=100
        )
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            y_preds = model.predict(testFeatureTensor)
            accuracy = ((y_preds == testLabelTensor).sum() / y_preds.shape[0]) * 100
            print(f"\nFinal Test Accuracy: {accuracy:.2f}%")
        
        # 4. Save model
        print("Step 4: Saving model...")
        
        # Save PyTorch model weights (recommended)
        model.save_model("ddos_classifier_model.pth")
        
        # Also save with pickle for compatibility with your Flask app
        with open("classificationModel.pkl", "wb") as f:
            pickle.dump(model, f)
        print("  ✓ Model saved as classificationModel.pkl")
        
        # Save transformer for preprocessing new data
        with open("transformer.pkl", "wb") as f:
            pickle.dump(transformer, f)
        print("  ✓ Transformer saved as transformer.pkl")
        
        # Save label mapping
        with open("label_mapping.pkl", "wb") as f:
            pickle.dump(loader.getLabelsMapping(), f)
        print("  ✓ Label mapping saved")
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"  ✗ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train()