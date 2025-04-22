import torch
import numpy as np

class TemporalEvaluator:
    def __init__(self, horizon_hours=48):
        self.horizon = horizon_hours
        
    def concordance_index(self, pred_risks, event_times, event_indicators):
        """Time-aware C-index calculation"""
        concordant = 0
        total = 0
        
        for i in range(len(pred_risks)):
            for j in range(i+1, len(pred_risks)):
                if event_indicators[i] and event_times[i] < event_times[j]:
                    if pred_risks[i] > pred_risks[j]:
                        concordant += 1
                    total += 1
                    
        return concordant / total if total > 0 else 0.5