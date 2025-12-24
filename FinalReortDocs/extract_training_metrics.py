"""
Extract training metrics from GRPO training log and save to CSV
"""
import re
import pandas as pd
import ast
from pathlib import Path

def extract_metrics_from_log(log_file_path):
    """
    Extract training metrics from the GRPO training log file.
    
    Args:
        log_file_path: Path to the log file
        
    Returns:
        DataFrame containing extracted metrics
    """
    metrics_list = []
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match metric dictionaries
    # Looking for lines that contain 'loss':, 'learning_rate':, etc.
    pattern = r"\{'loss':[^}]+\}"
    
    matches = re.findall(pattern, content)
    
    print(f"Found {len(matches)} training metric entries")
    
    for i, match in enumerate(matches):
        try:
            # Use ast.literal_eval to safely parse the dictionary string
            metrics_dict = ast.literal_eval(match)
            
            # Add step number (assuming metrics are logged every 10 steps)
            metrics_dict['step'] = (i + 1) * 10
            
            metrics_list.append(metrics_dict)
            
        except (ValueError, SyntaxError) as e:
            print(f"Warning: Failed to parse entry {i+1}: {e}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics_list)
    
    # Reorder columns to put step first
    if 'step' in df.columns:
        cols = ['step'] + [col for col in df.columns if col != 'step']
        df = df[cols]
    
    print(f"\nExtracted {len(df)} rows with {len(df.columns)} columns")
    print(f"Columns: {', '.join(df.columns[:10])}...")
    
    return df

def main():
    # Input and output paths
    log_file = Path("d:/PythonEx/cs566_group_project/output_archive/slurm-4979121-5-sft-grpo-train.out")
    output_csv = Path("d:/PythonEx/cs566_group_project/output_archive/training_metrics.csv")
    
    print(f"Reading log file: {log_file}")
    
    # Extract metrics
    df = extract_metrics_from_log(log_file)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Metrics saved to: {output_csv}")
    
    # Display summary statistics
    print("\n📊 Summary Statistics:")
    print("=" * 60)
    
    key_metrics = ['step', 'loss', 'learning_rate', 'rewards/reward_func/mean', 
                   'kl', 'entropy', 'grad_norm', 'epoch']
    
    for col in key_metrics:
        if col in df.columns:
            if col == 'step':
                print(f"{col:25s}: {df[col].min():.0f} → {df[col].max():.0f}")
            elif col == 'learning_rate':
                print(f"{col:25s}: {df[col].min():.2e} → {df[col].max():.2e}")
            elif col == 'rewards/reward_func/mean':
                print(f"{'accuracy':25s}: {df[col].min():.2%} → {df[col].max():.2%} (mean: {df[col].mean():.2%})")
            else:
                print(f"{col:25s}: {df[col].min():.4f} → {df[col].max():.4f} (mean: {df[col].mean():.4f})")
    
    print("=" * 60)
    print(f"\n📁 Total rows exported: {len(df)}")

if __name__ == "__main__":
    main()
