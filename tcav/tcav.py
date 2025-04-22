import numpy as np
import torch

class TCAV:
    def __init__(self, model, concept_names):
        self.model = model
        self.concept_names = concept_names
        
    def compute_tcav(self, dataloader, concept_idx):
        self.model.eval()
        gradients = []
        
        for batch in dataloader:
            x, _ = batch
            x.requires_grad_(True)
            
            _, concept_activations = self.model(x, return_concepts=True)
            concept_act = concept_activations[:, :, concept_idx]
            
            grad = torch.autograd.grad(
                outputs=concept_act,
                inputs=x,
                grad_outputs=torch.ones_like(concept_act),
                create_graph=False
            )[0]
            
            gradients.append(grad.detach().numpy())
            
        return np.concatenate(gradients, axis=0).mean(axis=0)
    
    def interpret_concepts(self, dataloader):
        tcav_scores = {}
        for i, concept_name in enumerate(self.concept_names):
            tcav_scores[concept_name] = self.compute_tcav(dataloader, i)
        return tcav_scores