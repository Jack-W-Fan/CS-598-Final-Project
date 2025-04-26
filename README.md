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

Standard Evaluation

    python scripts/evaluate.py \
    --model models/classifier.pt \
    --output_dir results/classification

Temporal Prediction (t-CAV)

    # Default 48-hour horizon
    python scripts/train_model.py --temporal

    # Custom horizon (e.g., 24 hours)
    python scripts/train_model.py --temporal --horizon 24

## Expected Results

Example outputs:
Concept importance scores: Diabetes (0.142), Hypertension (0.098)
Prediction AUC: 0.68 (chronic disease) vs. 0.72 (48-hr horizon)

## References

Mincu et al. (2021). "Concept-Based Model Explanations for Electronic Health Records." ACM CHIL.
Synthea Dataset. MITRE Corporation.