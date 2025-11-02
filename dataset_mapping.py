#!/usr/bin/env python3
"""
dataset_mapping.py

Load and map all downloaded datasets from the dataset/ folder.
For each dataset, show:
- Dataset structure and schema
- Full statistics (total rows per split)
- 3 sample entries from each split (train, validation, test)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd


# Define label mappings for each dataset
LABEL_MAPPINGS = {
    'go_emotions': {
        'label_field': 'labels',  # Field name in the data
        'is_multilabel': True,    # Whether it's multi-label classification
        'names': ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
                 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 
                 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 
                 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 
                 'sadness', 'surprise', 'neutral']
    },
    'emotion': {
        'label_field': 'label',
        'is_multilabel': False,
        'names': ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    },
    'tweet_eval_emotion': {
        'label_field': 'label',
        'is_multilabel': False,
        'names': ['anger', 'joy', 'optimism', 'sadness']
    },
    'imdb': {
        'label_field': 'label',
        'is_multilabel': False,
        'names': ['negative', 'positive']
    },
    'yelp_review_full': {
        'label_field': 'label',
        'is_multilabel': False,
        'names': ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']
    }
}


def get_emotion_names(dataset_name: str, labels) -> str:
    """Convert numeric labels to emotion names."""
    if dataset_name not in LABEL_MAPPINGS:
        return str(labels)  # Return as-is if no mapping
    
    mapping = LABEL_MAPPINGS[dataset_name]
    names = mapping['names']
    
    if mapping['is_multilabel']:
        # Handle multi-label case (like GoEmotions)
        if isinstance(labels, list):
            emotion_names = [names[label] if 0 <= label < len(names) else f"unknown_{label}" for label in labels]
            return emotion_names
        else:
            return [names[labels] if 0 <= labels < len(names) else f"unknown_{labels}"]
    else:
        # Handle single-label case
        if isinstance(labels, int) and 0 <= labels < len(names):
            return names[labels]
        else:
            return f"unknown_{labels}"


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file and return list of dictionaries."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return data


def analyze_dataset_structure(data: List[Dict[str, Any]], dataset_name: str, max_samples: int = 3) -> Dict[str, Any]:
    """Analyze the structure of a dataset split."""
    if not data:
        return {"total_rows": 0, "columns": [], "sample_types": {}, "samples": []}
    
    # Get column names from the first record
    columns = list(data[0].keys()) if data else []
    
    # Analyze data types from first few records
    sample_types = {}
    for col in columns:
        sample_values = [record.get(col) for record in data[:10] if col in record]
        if sample_values:
            sample_type = type(sample_values[0]).__name__
            # Check if it's consistently the same type
            consistent = all(type(val).__name__ == sample_type for val in sample_values)
            sample_types[col] = sample_type if consistent else "mixed"
    
    # Get sample entries with emotion names
    samples = []
    for record in data[:max_samples]:
        enhanced_record = record.copy()
        
        # Add emotion names if this dataset has label mappings
        if dataset_name in LABEL_MAPPINGS:
            mapping = LABEL_MAPPINGS[dataset_name]
            label_field = mapping['label_field']
            
            if label_field in enhanced_record:
                labels = enhanced_record[label_field]
                emotion_names = get_emotion_names(dataset_name, labels)
                enhanced_record['emotion_names'] = emotion_names
        
        samples.append(enhanced_record)
    
    return {
        "total_rows": len(data),
        "columns": columns,
        "sample_types": sample_types,
        "samples": samples
    }


def map_dataset_folder(dataset_dir: str) -> Dict[str, Any]:
    """Map a single dataset folder containing train/validation/test JSONL files."""
    dataset_path = Path(dataset_dir)
    dataset_name = dataset_path.name
    
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    splits_info = {}
    total_rows = 0
    
    # Look for standard split files
    split_files = {
        'train': dataset_path / 'train.jsonl',
        'validation': dataset_path / 'validation.jsonl',
        'test': dataset_path / 'test.jsonl',
        'unsupervised': dataset_path / 'unsupervised.jsonl'  # for IMDB
    }
    
    for split_name, split_file in split_files.items():
        if split_file.exists():
            print(f"\n--- {split_name.upper()} SPLIT ---")
            data = load_jsonl(str(split_file))
            analysis = analyze_dataset_structure(data, dataset_name)
            splits_info[split_name] = analysis
            total_rows += analysis['total_rows']
            
            print(f"Rows: {analysis['total_rows']:,}")
            print(f"Columns: {', '.join(analysis['columns'])}")
            print(f"Types: {analysis['sample_types']}")
            
            # Show label mapping info if available
            if dataset_name in LABEL_MAPPINGS:
                mapping = LABEL_MAPPINGS[dataset_name]
                print(f"Label mapping: {mapping['label_field']} -> {mapping['names']}")
                print(f"Multi-label: {mapping['is_multilabel']}")
            
            print(f"\nSample entries from {split_name}:")
            for i, sample in enumerate(analysis['samples'], 1):
                print(f"  {i}. {sample}")
    
    dataset_summary = {
        'name': dataset_name,
        'total_rows': total_rows,
        'splits': splits_info,
        'splits_available': list(splits_info.keys())
    }
    
    print(f"\nDATASET SUMMARY:")
    print(f"Total rows across all splits: {total_rows:,}")
    print(f"Splits available: {', '.join(splits_info.keys())}")
    
    return dataset_summary


def map_all_datasets(base_dir: str = 'dataset') -> Dict[str, Any]:
    """Map all datasets in the base directory."""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Dataset directory '{base_dir}' not found!")
        return {}
    
    print(f"MAPPING ALL DATASETS IN: {base_path.absolute()}")
    print(f"{'='*80}")
    
    all_datasets = {}
    total_datasets = 0
    total_rows_all = 0
    
    # Find all dataset subdirectories
    for dataset_dir in base_path.iterdir():
        if dataset_dir.is_dir():
            total_datasets += 1
            dataset_summary = map_dataset_folder(str(dataset_dir))
            all_datasets[dataset_summary['name']] = dataset_summary
            total_rows_all += dataset_summary['total_rows']
    
    # Overall summary
    print(f"\n{'='*80}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"Total datasets: {total_datasets}")
    print(f"Total rows across all datasets: {total_rows_all:,}")
    
    print(f"\nDataset breakdown:")
    for name, summary in all_datasets.items():
        splits = ', '.join(summary['splits_available'])
        print(f"  {name}: {summary['total_rows']:,} rows ({splits})")
    
    return all_datasets


def save_mapping_report(all_datasets: Dict[str, Any], output_file: str = 'dataset_mapping_report.json'):
    """Save the complete mapping report to a JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_datasets, f, indent=2, ensure_ascii=False)
        print(f"\nMapping report saved to: {output_file}")
    except Exception as e:
        print(f"Error saving report: {e}")


def main():
    """Main function to map all datasets."""
    print("DATASET MAPPING TOOL")
    print("===================")
    print("This tool maps all downloaded datasets and shows structure + samples\n")
    
    # Map all datasets
    all_datasets = map_all_datasets()
    
    if all_datasets:
        # Save detailed report
        save_mapping_report(all_datasets)
        
        print(f"\n{'='*80}")
        print("MAPPING COMPLETE!")
        print(f"{'='*80}")
        print("Use this information to understand dataset structures for training.")
        print("Key observations:")
        print("- GoEmotions: Good for multi-label emotion classification")
        print("- Emotion: Good for basic emotion classification") 
        print("- IMDB: Good for sentiment analysis")
        print("- TweetEval: Good for social media emotion patterns")
        print("- Yelp: Good for review sentiment analysis")
    else:
        print("No datasets found! Run dataset.py first to download datasets.")


if __name__ == "__main__":
    main()