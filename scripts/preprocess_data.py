from tcav.data_processor import SyntheaDataProcessor
import numpy as np
import os
from sklearn.model_selection import train_test_split
import argparse

def preprocess_and_save(data_dir, output_dir, temporal=False, horizon_hours=48):
    os.makedirs(output_dir, exist_ok=True)
    
    processor = SyntheaDataProcessor(
        patients_path=os.path.join(data_dir, 'patients_clean.csv'),
        conditions_path=os.path.join(data_dir, 'conditions.csv'),
        encounters_path=os.path.join(data_dir, 'encounters.csv')
    )
    
    if temporal:
        # Temporal mode processing
        sequences, event_times, event_indicators = processor.create_temporal_dataset(
            max_length=50, 
            horizon_hours=horizon_hours
        )
        
        # Split data
        train_seq, test_seq, train_events, test_events = train_test_split(
            sequences, event_indicators, test_size=0.2, random_state=42
        )
        
        # Save temporal files
        np.save(os.path.join(output_dir, 'train_sequences.npy'), train_seq)
        np.save(os.path.join(output_dir, 'train_events.npy'), train_events)
        np.save(os.path.join(output_dir, 'test_sequences.npy'), test_seq)
        np.save(os.path.join(output_dir, 'test_events.npy'), test_events)
        print(f"Saved temporal preprocessed data to {output_dir}")
    else:
        # Original classification processing
        sequences, labels = processor.create_dataset(max_length=50)
        
        train_seq, test_seq, train_lbl, test_lbl = train_test_split(
            sequences, labels, test_size=0.2, random_state=42
        )

        np.save(os.path.join(output_dir, 'train_sequences.npy'), train_seq)
        np.save(os.path.join(output_dir, 'train_labels.npy'), train_lbl)
        np.save(os.path.join(output_dir, 'test_sequences.npy'), test_seq)
        np.save(os.path.join(output_dir, 'test_labels.npy'), test_lbl)
        print(f"Saved classification preprocessed data to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--temporal', action='store_true', 
                       help='Enable temporal prediction mode')
    parser.add_argument('--horizon', type=int, default=48,
                       help='Prediction horizon in hours (for temporal mode)')
    args = parser.parse_args()
    
    preprocess_and_save(
        data_dir='data',
        output_dir='processed_data',
        temporal=args.temporal,
        horizon_hours=args.horizon
    )