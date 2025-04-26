# Reproducing Concept-Based Model Explanations for EHRs

This project implements Temporal Concept Activation Vectors (t-CAVs) for interpretable chronic disease prediction using synthetic EHR data from Synthea. Adapted from Mincu et al.'s ICU-focused work, it demonstrates concept-based explanations for GRU models on outpatient records.

## Setup

Installation

    git clone https://github.com/Jack-W-Fan/CS-598-Final-Project.git
    cd CS-598-Final-Project
    pip install -r requirements.txt

Install Synthea Dataset Version 2 from here and place in /data:
https://synthea.mitre.org/downloads 

## Running Pipline

1. Preprocess Data

Default 48-hour prediction horizon

    python scripts/preprocess_data.py --horizon 48

Custom horizon (e.g., 24 hours)

    python scripts/preprocess_data.py --horizon 24

2. Train Model

Standard Mode

    python scripts/train_model.py

Temporal Mode

    python scripts/train_model.py --temporal --horizon 48

3. Evaluate
Standard Concept Analysis

    python scripts/evaluate.py --model_path models/standard_model.pt
    
Temporal Predictions

    python scripts/evaluate.py --temporal --model_path models/temporal_model.pt

## Expected Results

Results will show up in the results folder. See classification for standard evaluation results (should see concept_importance.png and concept_scores.txt). See temporal for the temporal prediction (t-CAV) results (should see risk_preidictions.csv, temporal_metrics.txt, and temporal_predictions.npy).

Example outputs:
Standard

    Concept importance scores: Diabetes (0.142), Hypertension (0.098)
    Prediction AUC: 0.68 (chronic disease) vs. 0.72 (48-hr horizon)

Temporal

    AUROC: 0.782
    Average Risk Score: 0.154
    Event Rate: 23.5%

## References

Mincu et al. (2021). "Concept-Based Model Explanations for Electronic Health Records." ACM CHIL.
Synthea Dataset. MITRE Corporation.