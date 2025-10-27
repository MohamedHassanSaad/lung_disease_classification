
import yaml
import torch
from model_architectures import ProposedCNN
from preprocessing import CTPreprocessor
from optimization_algorithms import HybridBeeBayesianOptimizer

def main():
    # Load configuration
    with open('configs/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = ProposedCNN(num_classes=4)  # Example for CXR with 4 classes
    
    # Initialize preprocessing
    preprocessor = CTPreprocessor()
    
    # Initialize optimizer
    optimizer = HybridBeeBayesianOptimizer(
        search_space=config['training'],
        objective_function=train_and_evaluate
    )
    
    # Run optimization
    best_hyperparams = optimizer.optimize()
    
    # Train final model with best hyperparameters
    final_model = train_final_model(best_hyperparams)
    
    # Save model
    torch.save(final_model.state_dict(), 'trained_models/proposed_cnn_hybrid/model.pth')

if __name__ == '__main__':
    main()
