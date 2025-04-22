import torch
import numpy as np
from tcav.models import ConceptGRU, EHRDataset
from tcav.tcav import TCAV
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import yaml
import argparse
import os

def load_model(model_path, config):
    model = ConceptGRU(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        concept_dim=config['model']['concept_dim'],
        output_dim=config['model']['output_dim']
    )
    model.load_state_dict(torch.load(model_path))
    return model

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--temporal', action='store_true', 
                       help='Enable temporal prediction evaluation mode')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = load_config(args.config)
    
    # Load test data - now handles both modes
    test_seq = np.load('processed_data/test_sequences.npy')
    if args.temporal:
        test_targets = np.load('processed_data/test_events.npy')  # For temporal mode
    else:
        test_targets = np.load('processed_data/test_labels.npy')  # Original mode
    
    test_dataset = EHRDataset(test_seq, test_targets)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Load model
    model = load_model(args.model, config)
    
    if args.temporal:
        # TEMPORAL MODE EVALUATION
        predictions = []
        with torch.no_grad():
            for x, _ in test_loader:
                pred = model(x, temporal=True)
                predictions.extend(pred.numpy().flatten())
        
        # Save predictions
        np.save(os.path.join(args.output_dir, 'temporal_predictions.npy'), predictions)
        
        # Calculate basic metrics
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(test_targets, predictions)
        
        with open(os.path.join(args.output_dir, 'temporal_metrics.txt'), 'w') as f:
            f.write(f"AUROC: {auroc:.4f}\n")
            f.write(f"Average risk score: {np.mean(predictions):.4f}\n")
        
        print(f"Temporal evaluation saved to {args.output_dir}")
    
    else:
        # ORIGINAL CONCEPT EVALUATION (unchanged)
        concept_names = ['cardiovascular', 'respiratory', 'diabetes', 'hypertension']
        tcav = TCAV(model, concept_names)
        tcav_scores = tcav.interpret_concepts(test_loader)
        
        avg_concept_importance = {k: np.mean(np.abs(v)) for k, v in tcav_scores.items()}
        
        with open(os.path.join(args.output_dir, 'concept_scores.txt'), 'w') as f:
            for concept, score in avg_concept_importance.items():
                f.write(f"{concept}: {score:.4f}\n")
        
        plt.figure(figsize=(10, 6))
        plt.bar(avg_concept_importance.keys(), avg_concept_importance.values())
        plt.title('Concept Importance Scores (t-CAV)')
        plt.ylabel('Average Absolute Gradient Magnitude')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'concept_importance.png'))
        plt.close()
        
        print(f"Concept analysis saved to {args.output_dir}")

if __name__ == '__main__':
    main()