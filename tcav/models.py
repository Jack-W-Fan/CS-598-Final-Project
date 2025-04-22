import torch
import torch.nn as nn

class ConceptGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, concept_dim, output_dim=1):
        super(ConceptGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.concept_projector = nn.Linear(hidden_dim, concept_dim)
        
        # Temporal prediction head
        self.temporal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, return_concepts=False, temporal=False):
        gru_out, _ = self.gru(x)
        concept_activations = self.concept_projector(gru_out)
        
        if temporal:
            # Temporal prediction mode
            time_pred = self.temporal_head(gru_out[:, -1, :])
            if return_concepts:
                return time_pred, concept_activations
            return time_pred
        else:
            # Original classification mode
            last_hidden = gru_out[:, -1, :]
            output = self.sigmoid(self.fc(last_hidden))
            if return_concepts:
                return output, concept_activations
            return output.view(-1, 1)

class EHRDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]), 
            torch.FloatTensor([self.labels[idx]]).squeeze(0)
        )