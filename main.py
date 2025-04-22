import yaml
from tcav.data_processor import SyntheaDataProcessor
from tcav.models import ConceptGRU, EHRDataset
from tcav.train import train_model, evaluate_model
from tcav.tcav import TCAV
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

def load_config(config_path="configs/default.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    # Initialize data processor
    processor = SyntheaDataProcessor(
        patients_path=config['data']['patients_path'],
        conditions_path=config['data']['conditions_path'],
        encounters_path=config['data']['encounters_path']
    )
    
    # Create dataset
    sequences, labels = processor.create_dataset(
        max_length=config['data']['max_sequence_length']
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=config['training']['test_size'], random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=config['training']['val_size'], random_state=42
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        EHRDataset(X_train, y_train),
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        EHRDataset(X_val, y_val),
        batch_size=config['training']['batch_size']
    )
    test_loader = DataLoader(
        EHRDataset(X_test, y_test),
        batch_size=config['training']['batch_size']
    )
    
    # Initialize model
    model = ConceptGRU(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        concept_dim=config['model']['concept_dim'],
        output_dim=config['model']['output_dim']
    )
    
    # Train model
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader,
        epochs=config['training']['epochs'],
        lr=config['training']['learning_rate']
    )
    
    # Evaluate
    test_accuracy, _, _ = evaluate_model(model, test_loader)
    
    # Interpret concepts with t-CAV
    tcav = TCAV(model, list(processor.concepts.keys()))
    tcav_scores = tcav.interpret_concepts(test_loader)
    
    # Visualize concept importance
    avg_concept_importance = {k: np.mean(np.abs(v)) for k, v in tcav_scores.items()}
    plt.bar(avg_concept_importance.keys(), avg_concept_importance.values())
    plt.title('Average Concept Importance (t-CAV scores)')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('concept_importance.png')
    plt.close()
    
    return {
        'test_accuracy': test_accuracy,
        'concept_importance': avg_concept_importance,
        'train_losses': train_losses,
        'val_losses': val_losses
    }

if __name__ == '__main__':
    results = main()
    print(results)