import torch
import torch.nn as nn
import numpy as np
from tcav.models import ConceptGRU, EHRDataset
from tcav.data_processor import SyntheaDataProcessor  # Added import
from tcav.train import train_model, evaluate_model
from torch.utils.data import DataLoader
import yaml
import argparse

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--model_output', default='models/trained_model.pt')
    parser.add_argument('--temporal', action='store_true', 
                       help='Enable temporal prediction mode')
    parser.add_argument('--horizon', type=int, default=48, 
                      help='Prediction horizon in hours (for temporal mode only)')
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Initialize data processor
    processor = SyntheaDataProcessor(
        patients_path='data/patients.csv',
        conditions_path='data/conditions.csv',
        encounters_path='data/encounters.csv'
    )
    
    # Load data with temporal option
    if args.temporal:
        train_seq, train_lbl, train_events = processor.create_dataset(
            max_length=50,
            temporal=True,
            horizon_hours=args.horizon
        )
        test_seq, test_lbl, test_events = processor.create_dataset(
            max_length=50,
            temporal=True,
            horizon_hours=args.horizon
        )
    else:
        train_seq, train_lbl = processor.create_dataset(max_length=50)
        test_seq, test_lbl = processor.create_dataset(max_length=50)
    
    # Create validation split
    val_size = config['training']['val_size']
    train_idx = int(len(train_seq) * (1 - val_size))
    
    # Create datasets
    if args.temporal:
        train_dataset = EHRDataset(train_seq[:train_idx], train_events[:train_idx])  # Use events as labels
        val_dataset = EHRDataset(train_seq[train_idx:], train_events[train_idx:])
        test_dataset = EHRDataset(test_seq, test_events)
    else:
        train_dataset = EHRDataset(train_seq[:train_idx], train_lbl[:train_idx])
        val_dataset = EHRDataset(train_seq[train_idx:], train_lbl[train_idx:])
        test_dataset = EHRDataset(test_seq, test_lbl)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size']
    )
    
    # Initialize model
    model = ConceptGRU(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        concept_dim=config['model']['concept_dim'],
        output_dim=config['model']['output_dim']
    )
    
    # Training loop
    if args.temporal:
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.BCELoss()
        
        for epoch in range(config['training']['epochs']):
            model.train()
            epoch_loss = 0
            for x, y in train_loader:
                optimizer.zero_grad()
                pred = model(x, temporal=True)
                loss = criterion(pred.squeeze(), y.squeeze())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    pred = model(x_val, temporal=True)
                    val_loss += criterion(pred.squeeze(), y_val.squeeze()).item()  # And this line
            
            print(f'Epoch {epoch+1}: Loss={epoch_loss/len(train_loader):.4f}, Val Loss={val_loss/len(val_loader):.4f}')

    else:
        # Original training
        train_model(
            model,
            train_loader,
            val_loader,
            epochs=config['training']['epochs'],
            lr=config['training']['learning_rate']
        )
    
    # Save model
    torch.save(model.state_dict(), args.model_output)
    print(f'Model saved to {args.model_output}')

if __name__ == '__main__':
    main()