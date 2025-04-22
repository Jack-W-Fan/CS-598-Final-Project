import pandas as pd
import numpy as np
from datetime import timedelta

class SyntheaDataProcessor:
    def __init__(self, patients_path, conditions_path, encounters_path):
        self.patients = pd.read_csv(patients_path, on_bad_lines='skip')
        self.encounters = pd.read_csv(
            encounters_path,
            usecols=['PATIENT', 'ID', 'DATE'],
            parse_dates=['DATE'],
            dtype={'PATIENT': 'category', 'ID': 'category'},
            on_bad_lines='skip'
        )
        self.conditions = pd.read_csv(
            conditions_path,
            usecols=['PATIENT', 'ENCOUNTER', 'CODE'],
            dtype={'PATIENT': 'category', 'ENCOUNTER': 'category', 'CODE': 'category'},
            on_bad_lines='skip'
        )
        
        self.concepts = {
            'cardiovascular': ['I20', 'I21', 'I22', 'I23', 'I24', 'I25'],
            'respiratory': ['J18', 'J20', 'J40', 'J45'],
            'diabetes': ['E10', 'E11'],
            'hypertension': ['I10']
        }

    def create_dataset(self, max_length=100, temporal=False, horizon_hours=48):
        merged = pd.merge(
            self.conditions,
            self.encounters,
            left_on=['PATIENT', 'ENCOUNTER'],
            right_on=['PATIENT', 'ID'],
            how='left'
        )
        merged['DATE'] = pd.to_datetime(merged['DATE'])
        merged = merged.sort_values(['PATIENT', 'DATE'])
        
        # Create concept indicators
        for name, codes in self.concepts.items():
            merged[name] = merged['CODE'].isin(codes).astype(int)
        
        concept_cols = list(self.concepts.keys())
        sequences = []
        
        if temporal:
            # Temporal mode processing
            event_indicators = []
            for patient_id, group in merged.groupby('PATIENT'):
                # Create hourly bins
                group = group.set_index('DATE')[concept_cols].resample('1H').max().fillna(0)
                concept_data = group.values
                
                # Pad sequence
                seq_length = min(len(concept_data), max_length)
                padded = np.zeros((max_length, len(self.concepts)))
                padded[-seq_length:] = concept_data[-seq_length:]
                sequences.append(padded)
                
                # Calculate event indicators
                time_deltas = (group.index[1:] - group.index[:-1]).total_seconds() / 3600
                has_event = (time_deltas <= horizon_hours).any()
                event_indicators.append(int(has_event))
            
            return np.array(sequences), None, np.array(event_indicators)
        
        else:
            # Standard classification mode
            labels = []
            for _, group in merged.groupby('PATIENT'):
                concept_matrix = group[concept_cols].values
                sequences.append(concept_matrix)
                labels.append(int(concept_matrix.sum() > 0))
            
            # Pad sequences
            padded_sequences = np.zeros((len(sequences), max_length, len(self.concepts)))
            for i, seq in enumerate(sequences):
                seq_length = min(len(seq), max_length)
                padded_sequences[i, -seq_length:, :] = seq[-seq_length:]
            
            return padded_sequences, np.array(labels)

    # def create_dataset(self, max_length=100):
    #     """Original classification dataset creation"""
    #     sequences, labels = self.preprocess_data()
        
    #     padded_sequences = np.zeros((len(sequences), max_length, len(self.concepts)))
    #     for i, seq in enumerate(sequences):
    #         seq_length = min(len(seq), max_length)
    #         padded_sequences[i, -seq_length:, :] = seq[-seq_length:]
            
    #     return padded_sequences, np.array(labels)

    def preprocess_data(self):
        """Original preprocessing logic"""
        merged = self.conditions.merge(
            self.encounters, 
            left_on=['PATIENT', 'ENCOUNTER'],
            right_on=['PATIENT', 'ID'],
            how='left'
        )
        merged['DATE'] = pd.to_datetime(merged['DATE'])
        merged = merged.sort_values(['PATIENT', 'DATE'])
        
        for concept_name, codes in self.concepts.items():
            merged[concept_name] = merged['CODE_x'].isin(codes).astype(int)
        
        sequences = []
        labels = []
        for _, group in merged.groupby('PATIENT'):
            concept_matrix = group[list(self.concepts.keys())].values
            sequences.append(concept_matrix)
            labels.append(int(concept_matrix.sum() > 0))
            
        return sequences, labels