#!/usr/bin/env python3
"""
dataset_cleaner.py

Remove unlabeled or invalid labeled entries from all datasets.
Specifically targets:
- Entries without label/labels fields
- Entries with null labels
- Entries with invalid label values (like -1 in IMDB unsupervised)
- Empty label lists
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any


# Define valid label ranges for each dataset
VALID_LABEL_CONFIGS = {
    'go_emotions': {'field': 'labels', 'valid_range': list(range(28)), 'is_list': True},
    'emotion': {'field': 'label', 'valid_range': list(range(6)), 'is_list': False},
    'tweet_eval_emotion': {'field': 'label', 'valid_range': list(range(4)), 'is_list': False},
    'imdb': {'field': 'label', 'valid_range': [0, 1], 'is_list': False},
    'yelp_review_full': {'field': 'label', 'valid_range': list(range(5)), 'is_list': False}
}


def is_valid_entry(data: Dict[str, Any], dataset_name: str) -> bool:
    """Check if a data entry has valid labels."""
    if dataset_name not in VALID_LABEL_CONFIGS:
        return True  # Don't filter unknown datasets
    
    config = VALID_LABEL_CONFIGS[dataset_name]
    label_field = config['field']
    valid_range = config['valid_range']
    is_list = config['is_list']
    
    # Check if label field exists
    if label_field not in data:
        return False
    
    label_value = data[label_field]
    
    # Check for null/None labels
    if label_value is None:
        return False
    
    if is_list:
        # Multi-label case (like GoEmotions)
        if not isinstance(label_value, list):
            return False
        if len(label_value) == 0:  # Empty list
            return False
        # All labels must be in valid range
        return all(label in valid_range for label in label_value)
    else:
        # Single-label case
        return label_value in valid_range


def clean_dataset_file(input_file: Path, output_file: Path, dataset_name: str) -> Dict[str, int]:
    """Clean a single JSONL file by removing invalid entries."""
    stats = {'total': 0, 'valid': 0, 'removed': 0}
    
    valid_entries = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                stats['total'] += 1
                
                if is_valid_entry(data, dataset_name):
                    valid_entries.append(data)
                    stats['valid'] += 1
                else:
                    stats['removed'] += 1
                    
            except json.JSONDecodeError as e:
                print(f"JSON decode error in {input_file} line {line_num}: {e}")
                stats['removed'] += 1
    
    # Write cleaned data
    os.makedirs(output_file.parent, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in valid_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    return stats


def clean_all_datasets(input_dir: str = 'dataset_original', output_dir: str = 'dataset') -> None:
    """Clean all datasets and save to output directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Input directory '{input_dir}' not found!")
        return
    
    print(f"CLEANING DATASETS")
    print(f"Input: {input_path.absolute()}")
    print(f"Output: {output_path.absolute()}")
    print("=" * 60)
    
    total_stats = {'total': 0, 'valid': 0, 'removed': 0}
    
    for dataset_folder in input_path.iterdir():
        if dataset_folder.is_dir():
            dataset_name = dataset_folder.name
            print(f"\nCleaning dataset: {dataset_name}")
            
            dataset_stats = {'total': 0, 'valid': 0, 'removed': 0}
            
            for split_file in dataset_folder.glob('*.jsonl'):
                split_name = split_file.stem
                
                # Skip splits that are known to be unlabeled
                if split_name == 'unsupervised' and dataset_name == 'imdb':
                    print(f"  Skipping {split_name} (unsupervised data)")
                    continue
                
                output_file = output_path / dataset_name / f"{split_name}.jsonl"
                
                stats = clean_dataset_file(split_file, output_file, dataset_name)
                
                print(f"  {split_name}: {stats['valid']}/{stats['total']} valid entries ({stats['removed']} removed)")
                
                # Update totals
                for key in stats:
                    dataset_stats[key] += stats[key]
                    total_stats[key] += stats[key]
            
            if dataset_stats['total'] > 0:
                removal_rate = (dataset_stats['removed'] / dataset_stats['total']) * 100
                print(f"  Dataset total: {dataset_stats['valid']}/{dataset_stats['total']} valid ({removal_rate:.1f}% removed)")
    
    print(f"\n" + "=" * 60)
    print(f"CLEANING SUMMARY")
    print(f"Total entries processed: {total_stats['total']:,}")
    print(f"Valid entries kept: {total_stats['valid']:,}")
    print(f"Invalid entries removed: {total_stats['removed']:,}")
    
    if total_stats['total'] > 0:
        removal_rate = (total_stats['removed'] / total_stats['total']) * 100
        print(f"Overall removal rate: {removal_rate:.2f}%")
    
    print(f"\nCleaned datasets saved to: {output_path.absolute()}")


def main():
    """Main function."""
    print("DATASET CLEANER")
    print("===============")
    print("This tool removes unlabeled or invalid labeled entries from datasets.\n")
    
    # Clean datasets
    clean_all_datasets()
    
    print("\nCleaning complete!")
    print("\nTo use cleaned datasets:")
    print("1. Backup original: mv dataset dataset_original")
    print("2. Use cleaned: mv dataset_cleaned dataset")
    print("3. Or update your scripts to use dataset_cleaned/")


if __name__ == "__main__":
    main()