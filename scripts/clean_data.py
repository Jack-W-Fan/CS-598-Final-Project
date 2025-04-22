import pandas as pd
from pathlib import Path

def clean_patients_file(input_path, output_path):
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    # Keep only properly formatted lines (17 columns)
    cleaned_lines = [lines[0]]  # Keep header
    for line in lines[1:]:
        if line.count(',') == 16:  # 17 columns
            cleaned_lines.append(line)
    
    with open(output_path, 'w') as f:
        f.writelines(cleaned_lines)

if __name__ == '__main__':
    data_dir = Path('data')
    clean_patients_file(data_dir/'patients.csv', data_dir/'patients_clean.csv')
    print("Created cleaned patients file at data/patients_clean.csv")